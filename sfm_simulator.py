import numpy as np

class SFMEnvironment:
    """
    基于扩展 Store-and-Forward Modeling (SFM) 的连续-离散混合系统物理仿真器
    引入了严格的质量守恒(Mass Conservation)和节点供需模型(Demand-Supply)
    """
    def __init__(self, links, junctions, T=60, noise_std=0.0):
        """
        :param links: dict, 路段属性
        :param junctions: dict, 节点属性
        :param T: int, 控制步长 (秒)
        :param noise_std: float, 客观世界的人群流率扰动程度 (测试鲁棒性用)
        """
        self.links = links
        self.junctions = junctions
        self.T = T
        self.noise_std = noise_std  # 增加现实物理世界的随机扰动
        self.num_links = len(links)
        self.link_map = {l_id: idx for idx, l_id in enumerate(links.keys())}
        
    def get_turning_proportions(self, delta):
        # 此部分与之前完全一致
        tau = {}
        for j_id, j_info in self.junctions.items():
            in_links = j_info['in_links']
            out_links = j_info['out_links']
            if not in_links or not out_links:
                continue
            if j_info['type'] == 'standard' or len(out_links) == 1:
                for w in in_links:
                    tau[(w, out_links[0])] = 1.0
            elif j_info['type'] == 'split':
                choice = delta.get(j_id, 0)
                for w in in_links:
                    if choice == 0:
                        for z in out_links:
                            tau[(w, z)] = 1.0 / len(out_links)
                    else:
                        target_out = out_links[choice - 1]
                        for z in out_links:
                            tau[(w, z)] = 1.0 if z == target_out else 0.0
        return tau

    def step(self, x_current, u, delta, demand):
        """
        执行一步物理仿真 (严格保障质量守恒)
        """
        x_next = np.copy(x_current)
        P_tentative = np.zeros(self.num_links) 
        Q_tentative = np.zeros(self.num_links) 
        
        tau = self.get_turning_proportions(delta)
        
        # ---------------------------------------------------------
        # 第一阶段：计算上游的“流出需求 (Tentative Outflow)”
        # ---------------------------------------------------------
        for l_id, l_info in self.links.items():
            idx = self.link_map[l_id]
            end_node = l_info['end_node']
            u_j = u.get(end_node, 1.0) 
            
            # 加入客观世界的参数扰动 (Parameter Mismatch for Robustness test)
            actual_outflow_rate = l_info['max_outflow'] * max(0.0, np.random.normal(1.0, self.noise_std))
            
            max_volume = self.T * actual_outflow_rate * u_j
            P_tentative[idx] = min(max_volume, x_current[idx])

        # 根据分配比例，计算下游的“流入需求 (Tentative Inflow)”
        for (w, z), ratio in tau.items():
            w_idx = self.link_map[w]
            z_idx = self.link_map[z]
            Q_tentative[z_idx] += ratio * P_tentative[w_idx]
            
        # ---------------------------------------------------------
        # 第二阶段：计算下游的“接收能力 (Supply)” 及“节点拥堵反压”
        # ---------------------------------------------------------
        alpha = np.ones(self.num_links) # 接收比例 (Acceptance Ratio)
        
        for l_id, l_info in self.links.items():
            idx = self.link_map[l_id]
            C_z = l_info['capacity']
            
            # 可用空间 = 容量 - 当前人数 + 预期流出量(腾出的空间)
            available_space = max(0, C_z - x_current[idx] + P_tentative[idx])
            
            # 如果期望流入量超出了物理容量，计算接收比例 alpha < 1.0
            if Q_tentative[idx] > available_space and Q_tentative[idx] > 0:
                alpha[idx] = available_space / Q_tentative[idx]
                
        # ---------------------------------------------------------
        # 第三阶段：根据接收比例，计算真正的流入流出 (True Q and P)
        # ---------------------------------------------------------
        P_actual = np.zeros(self.num_links)
        Q_actual = np.zeros(self.num_links)
        
        for l_id, l_info in self.links.items():
            w_idx = self.link_map[l_id]
            
            # 找到当前路段流向的所有下游路段，取它们之中最拥堵的 alpha
            # (交通流理论中的 FIFO 原则：一个出口堵了，后面全堵)
            min_alpha = 1.0
            for z_id in self.links.keys():
                if (l_id, z_id) in tau and tau[(l_id, z_id)] > 0:
                    z_idx = self.link_map[z_id]
                    if alpha[z_idx] < min_alpha:
                        min_alpha = alpha[z_idx]
                        
            # 真实流出量等于预期流出量乘以下游的最小接收比例
            P_actual[w_idx] = P_tentative[w_idx] * min_alpha
            
        # 根据真实的流出量，重新计算真实的流入量，确保质量绝对守恒
        for (w, z), ratio in tau.items():
            w_idx = self.link_map[w]
            z_idx = self.link_map[z]
            Q_actual[z_idx] += ratio * P_actual[w_idx]

        # ---------------------------------------------------------
        # 第四阶段：状态更新
        # ---------------------------------------------------------
        for idx in range(self.num_links):
            # demand[idx] 代表这一步中涌入的外部需求绝对人数
            actual_demand = demand[idx] * max(0.0, np.random.normal(1.0, self.noise_std))
            
            x_next[idx] = x_current[idx] + Q_actual[idx] - P_actual[idx] + actual_demand
            x_next[idx] = max(0.0, x_next[idx]) # 防止浮点误差
            
        return x_next, Q_actual, P_actual