import gurobipy as gp
from gurobipy import GRB
import numpy as np

class GurobiMPCController:
    """
    基于 Gurobi 的混合整数二次规划 (MIQP) 预测控制器
    对应论文中的 Eq. 4.1 和 Eq. 4.2
    """
    def __init__(self, links, junctions, T=60, Np=5, weight_R=1e-5):
        """
        :param links: 路段属性字典
        :param junctions: 节点属性字典
        :param T: 控制步长
        :param Np: 预测时域 (Prediction Horizon, 默认5)
        :param weight_R: 控制信号的惩罚权重 (对应论文中的 R = 10^-5 * I)
        """
        self.links = links
        self.junctions = junctions
        self.T = T
        self.Np = Np
        self.weight_R = weight_R
        self.num_links = len(links)
        self.link_map = {l_id: idx for idx, l_id in enumerate(links.keys())}

        # 用于存储上一个时间步的最优控制序列，作为本次的 MIP Start
        self.last_optimal_u = None
        self.last_optimal_delta = None
        
    def get_split_choices(self, j_id):
        """
        动态获取可用分流模式 (泛化适配任意度数的网络)
        0: 均分 (所有出路平分流量)
        1 ~ N: 将所有流量全量导向第 i 条出路
        """
        num_out = len(self.junctions[j_id]['out_links'])
        return list(range(num_out + 1)) # 返回 [0, 1, ..., num_out]
        
    def get_tau(self, j_info, choice):
        """动态生成分配比例字典"""
        tau = {}
        in_links = j_info['in_links']
        out_links = j_info['out_links']
        
        for w in in_links:
            if choice == 0:  # 0代表均分策略
                for z in out_links:
                    tau[(w, z)] = 1.0 / len(out_links)
            else:            # 1~N 代表全量导向某一条特定分支
                target_out = out_links[choice - 1]
                for z in out_links:
                    tau[(w, z)] = 1.0 if z == target_out else 0.0
        return tau

    def optimize(self, x_current, demand_forecast):
        """
        求解 MIQP 并返回下一步的最优控制动作
        :param x_current: np.array, 当前各路段实际人数
        :param demand_forecast: np.array, 形状为 (Np, num_links) 的未来外部需求预测
        :return: (best_u, best_delta) 控制字典
        """
        # 创建环境，关闭打印输出以加快滚动优化的速度
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0) 
        # ====================================================
        env.setParam("TimeLimit", 60.0)  # 强制单步求解不超过 60 秒
        env.setParam("MIPGap", 0.05)     # 允许 5% 的最优性容差，瞬间剪断海量废枝
        env.setParam("MIPFocus", 1)      # 策略：不钻牛角尖，侧重于快速寻找可行解
        env.setParam("Heuristics", 0.5)  # 动用 50% 的算力执行启发式搜索，快速打通安全逃生通道
        env.setParam("Seed", 42)         # 强制锁定 Gurobi 内部的分支定界随机种子
        # ====================================================
        env.start()
        model = gp.Model("Hybrid_MPC_Crowd_Control", env=env)

        # ---------------------------------------------------------
        # 1. 声明决策变量
        # ---------------------------------------------------------
        # x[t, l]: 时间步 t 时路段 l 的人数 (连续变量)
        x = {}
        for t in range(self.Np + 1):
            for l_id, l_info in self.links.items():
                idx = self.link_map[l_id]
                # 物理硬约束 Eq. 3.9: 0 <= x_z(k) <= x_z,max
                x[t, idx] = model.addVar(lb=0.0, ub=l_info['capacity'], name=f"x_{t}_{idx}")
                
        # P[t, l]: 时间步 t 时路段 l 的实际流出量 (连续变量)
        P = {}
        for t in range(self.Np):
            for l_id in self.links.keys():
                idx = self.link_map[l_id]
                P[t, idx] = model.addVar(lb=0.0, name=f"P_{t}_{idx}")

        # Flow[t, w, z]: 从路段 w 流向路段 z 的流量 (连续变量)
        Flow = {}
        for t in range(self.Np):
            for j_id, j_info in self.junctions.items():
                for w in j_info['in_links']:
                    for z in j_info['out_links']:
                        w_idx, z_idx = self.link_map[w], self.link_map[z]
                        Flow[t, w_idx, z_idx] = model.addVar(lb=0.0, name=f"F_{t}_{w_idx}_{z_idx}")

        # u[t, j]: 时间步 t 节点 j 的标准红绿灯信号 (二进制 0-1)
        u = {}
        for t in range(self.Np):
            for j_id in self.junctions.items():
                u[t, j_id[0]] = model.addVar(vtype=GRB.BINARY, name=f"u_{t}_{j_id[0]}")

        # delta_bin[t, j, choice]: 分流节点的离散选择 (二进制 0-1)
        delta_bin = {}
        for t in range(self.Np):
            for j_id, j_info in self.junctions.items():
                if j_info['type'] == 'split':
                    for c in self.get_split_choices(j_id):
                        delta_bin[t, j_id, c] = model.addVar(vtype=GRB.BINARY, name=f"d_{t}_{j_id}_{c}")

        # 引入纯线性辅助流量变量 P_split
        # P_split[t, w, c]: 当路口选择了模式 c 时，从路段 w 流出的实际流量
        P_split = {}
        for t in range(self.Np):
            for j_id, j_info in self.junctions.items():
                if j_info['type'] == 'split':
                    choices = self.get_split_choices(j_id)
                    for w in j_info['in_links']:
                        w_idx = self.link_map[w]
                        # 物理上限就是该路的最大流出量，系数极其紧凑安全！
                        max_out_w = self.T * self.links[w]['max_outflow']
                        for c in choices:
                            P_split[t, w_idx, c] = model.addVar(lb=0.0, ub=max_out_w, name=f"Ps_{t}_{w_idx}_{c}")

        # 注入 MIP Starts (热启动)
        if self.last_optimal_u is not None and self.last_optimal_delta is not None:
            for t in range(self.Np - 1): # 将 t+1 的解作为 t 的初始解
                for j_id in self.junctions.keys():
                    u[t, j_id].Start = self.last_optimal_u[t + 1, j_id]
                    if self.junctions[j_id]['type'] == 'split':
                        for c in self.get_split_choices(j_id):
                            delta_bin[t, j_id, c].Start = self.last_optimal_delta[t + 1, j_id, c]
            
            # 对于序列的最后一步 (Np-1)，保守起见复制前一步的决策
            for j_id in self.junctions.keys():
                u[self.Np - 1, j_id].Start = self.last_optimal_u[self.Np - 1, j_id]
                if self.junctions[j_id]['type'] == 'split':
                    for c in self.get_split_choices(j_id):
                        delta_bin[self.Np - 1, j_id, c].Start = self.last_optimal_delta[self.Np - 1, j_id, c]


        # ---------------------------------------------------------
        # 2. 状态方程与约束配置
        # ---------------------------------------------------------
        # 初始状态约束 (x_{k|k} = x(k))
        for l_id, l_info in self.links.items():
            idx = self.link_map[l_id]
            safe_initial_x = min(x_current[idx], l_info['capacity'])
            model.addConstr(x[0, idx] == safe_initial_x, name=f"init_{idx}")

        for t in range(self.Np):
            # A. 限制实际流出量 P_z(k) 
            for l_id, l_info in self.links.items():
                idx = self.link_map[l_id]
                end_node = l_info['end_node']
                
                # P <= 当前路段内存量
                model.addConstr(P[t, idx] <= x[t, idx])
                
                # P <= 信号灯许可的最大通行能力
                # 如果是出口路段(无下游节点)，则默认畅通
                if end_node in self.junctions:
                    max_cap = self.T * l_info['max_outflow']
                    model.addConstr(P[t, idx] <= max_cap * u[t, end_node])
                else:
                    model.addConstr(P[t, idx] <= self.T * l_info['max_outflow'])

            # B. 节点流量分配逻辑 (替代原本的 implies)
            for j_id, j_info in self.junctions.items():
                if not j_info['in_links'] or not j_info['out_links']:
                    continue
                if j_info['type'] == 'standard' or len(j_info['out_links']) == 1:
                    # 标准节点：流入等于流出 (比例恒为1)
                    for w in j_info['in_links']:
                        w_idx = self.link_map[w]
                        out_l = j_info['out_links'][0]
                        z_idx = self.link_map[out_l]
                        model.addConstr(Flow[t, w_idx, z_idx] == P[t, w_idx])
                elif j_info['type'] == 'split':
                    # 混合逻辑：必须且只能选择一种模式
                    choices = self.get_split_choices(j_id)
                    model.addConstr(gp.quicksum(delta_bin[t, j_id, c] for c in choices) == 1)

                    for w in j_info['in_links']:
                        w_idx = self.link_map[w]
                        max_out_w = self.T * self.links[w]['max_outflow']
                        
                        # 1. 各个模式下的流量之和，必须等于路段总流出量 P
                        model.addConstr(gp.quicksum(P_split[t, w_idx, c] for c in choices) == P[t, w_idx])
                        
                        # 2. 只有被选中的模式 (delta=1)，对应的 P_split 才能有流量
                        for c in choices:
                            model.addConstr(P_split[t, w_idx, c] <= max_out_w * delta_bin[t, j_id, c])
                            
                        # 3. 计算最终流向各个下游 z 的 Flow
                        for z in j_info['out_links']:
                            z_idx = self.link_map[z]
                            flow_expr = gp.LinExpr()
                            for c in choices:
                                tau = self.get_tau(j_info, c)
                                ratio = tau.get((w, z), 0.0)
                                if ratio > 0:
                                    flow_expr += ratio * P_split[t, w_idx, c]
                            
                            # 绝对的线性等式，彻底告别数值不稳定！
                            model.addConstr(Flow[t, w_idx, z_idx] == flow_expr)
                                                            
            # C. 状态转移更新 Eq. 3.12 (质量守恒)
            for l_id in self.links.keys():
                idx = self.link_map[l_id]
                # Q_in = 所有上游流过来的流量之和
                Q_in = gp.quicksum(Flow[t, w_idx, idx] 
                                   for w_idx in range(self.num_links) 
                                   if (t, w_idx, idx) in Flow)
                
                model.addConstr(x[t+1, idx] == x[t, idx] + Q_in - P[t, idx] + demand_forecast[t, idx])

        # ---------------------------------------------------------
        # 3. 目标函数配置 (Eq. 4.2)
        # ---------------------------------------------------------
        obj_expr = gp.QuadExpr()
        
        # 惩罚路段占有率，Q = diag(1 / x_{z,max}) 对应论文配置
        for t in range(1, self.Np + 1):
            for l_id, l_info in self.links.items():
                idx = self.link_map[l_id]
                if 'exit' in l_id: pass
                elif 'entry' in l_id:
                    # 内部路段在 100% 满载时的最大边际惩罚为 2.0。
                    # 将入口推力设为 1.7，MPC 会在内部路段达到 85% 容量时主动停止推入 (1.7 / 2.0 = 0.85)。
                    # 这精准地预留了 15% 的物理安全缓冲，完美吸收 sfm_simulator 中 15% 的随机环境噪声！
                    obj_expr += 1.7 * x[t, idx]
                else:
                    weight_Q = 1.0 / max(1.0, l_info['capacity'])
                    obj_expr += weight_Q * (x[t, idx] * x[t, idx])
                
        # 惩罚信号灯变动，减少过度控制
        for t in range(self.Np):
            for j_id, j_info in self.junctions.items():
                obj_expr += self.weight_R * (u[t, j_id] * u[t, j_id])
                if j_info['type'] == 'split':
                    for c in self.get_split_choices(j_id):
                        if c > 0:
                            obj_expr += 0.5 * delta_bin[t, j_id, c]
                
        model.setObjective(obj_expr, GRB.MINIMIZE)

        # ---------------------------------------------------------
        # 4. 求解与异常处理 (Infeasible Issues 对应论文的容错逻辑)
        # ---------------------------------------------------------
        model.optimize()
        
        best_u = {}
        best_delta = {}
        
        if model.SolCount > 0:
            # 记录整个 Np 序列，留作下一步的 MIP Start
            self.last_optimal_u = {}
            self.last_optimal_delta = {}
            
            for t_step in range(self.Np):
                for j_id, j_info in self.junctions.items():
                    self.last_optimal_u[t_step, j_id] = round(u[t_step, j_id].X)
                    if j_info['type'] == 'split':
                        for c in self.get_split_choices(j_id):
                            if delta_bin[t_step, j_id, c].X > 0.5:
                                self.last_optimal_delta[t_step, j_id, c] = 1
                            else:
                                self.last_optimal_delta[t_step, j_id, c] = 0

            # 仅提取第一时间步 t=0 的控制信号给当前环境执行
            for j_id, j_info in self.junctions.items():
                best_u[j_id] = self.last_optimal_u[0, j_id]
                if j_info['type'] == 'split':
                    for c in self.get_split_choices(j_id):
                        if self.last_optimal_delta[0, j_id, c] == 1:
                            best_delta[j_id] = c
                            break
        else:
            # 遭遇不可行解 (Infeasible Issues)，触发 Default fallback 策略
            # 对应论文 Algorithm 1 第 21-24 行的容错机制
            print(f"Warning: MPC Infeasible or Timeout! Applying default strategy.")
            self.last_optimal_u = None
            self.last_optimal_delta = None

            for j_id, j_info in self.junctions.items():
                best_u[j_id] = 1.0 # 默认全部放行
                # 但如果是外部源头汇入点 (L_entry 的终点)，坚决亮红灯，把外部洪峰挡住！
                for in_l in j_info['in_links']:
                    if 'entry' in in_l:
                        best_u[j_id] = 0.0  # 边界截流死死守住
                        break
                if j_info['type'] == 'split':
                    best_delta[j_id] = 0 # 默认导向第一条路
                    
        return best_u, best_delta