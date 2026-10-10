import numpy as np
import copy

class GAController:
    """
    进化版遗传算法控制器 (Evolved GA Controller)
    引入了软惩罚 (Soft Penalty)、精英保留 (Elitism) 和启发式初始化
    """
    def __init__(self, links, junctions, T=60, Np=5, pop_size=50, max_iter=30, weight_R=1e-5):
        self.links = links
        self.junctions = junctions
        self.T = T
        self.Np = Np
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.weight_R = weight_R
        
        self.num_links = len(links)
        self.link_map = {l_id: idx for idx, l_id in enumerate(links.keys())}
        
        self.weight_Q = np.zeros(self.num_links)
        for l_id, l_info in self.links.items():
            idx = self.link_map[l_id]
            
            if 'exit' in l_id:
                self.weight_Q[idx] = 0.0
            elif 'entry' in l_id:
                self.weight_Q[idx] = 2.0
            else:
                self.weight_Q[idx] = 1.0 / max(1.0, l_info['capacity'])
            
        self.split_junctions = [j_id for j_id, j_info in junctions.items() if j_info['type'] == 'split']
        self.num_split = len(self.split_junctions)
        self.num_nodes = len(self.junctions)
        
        # 记忆上一次优化的最佳结果，用于“启发式初始化”
        self.prev_best_delta = None
        self.prev_best_u = None

        # 找出网络中最大的岔路口分支数，以确定 GA 基因(delta)的上限
        max_out_degree = max([len(j_info['out_links']) for j_info in self.junctions.values() if j_info['type'] == 'split'], default=2)
        # 基因取值范围是 [0, max_out_degree]
        self.max_choice = max_out_degree + 1

    def _get_tau(self, j_info, choice):
        """简单的比例分配辅助函数（含越界保护映射）"""
        tau = {}
        in_links = j_info['in_links']
        out_links = j_info['out_links']
        num_out = len(out_links)
        
        # 如果 GA 随机生成的 choice 超出了当前路口的实际出路数量 (num_out)，
        # 取模运算会将其均匀地映射回合法的 [0, num_out] 范围内。
        # 例如 num_out = 2，可选合法 choice 为 {0, 1, 2}
        # 当传入 choice = 3 时，3 % 3 = 0 (映射回均分策略)
        # 当传入 choice = 4 时，4 % 3 = 1 (映射回走第一条路)
        valid_choice = choice % (num_out + 1)
        
        for w in in_links:
            if valid_choice == 0:
                for z in out_links: tau[(w, z)] = 1.0 / num_out
            else:
                target = out_links[valid_choice - 1]
                for z in out_links: tau[(w, z)] = 1.0 if z == target else 0.0
        return tau

    def _evaluate_fitness(self, individual_delta, individual_u, x_current, demand_forecast):
        """评估单个染色体的适应度 (越小越好)"""
        x_pred = np.copy(x_current)
        fitness = 0.0
        
        for t in range(self.Np):
            current_delta = {j_id: individual_delta[t, i] for i, j_id in enumerate(self.split_junctions)}
            current_u = {j_id: individual_u[t, i] for i, j_id in enumerate(self.junctions.keys())}
            
            P = np.zeros(self.num_links)
            Q = np.zeros(self.num_links)
            
            # 1. 计算近似流出
            for l_id, l_info in self.links.items():
                idx = self.link_map[l_id]
                u_val = current_u.get(l_info['end_node'], 1.0)
                P[idx] = min(self.T * l_info['max_outflow'] * u_val, x_pred[idx])
                
            # 2. 计算近似流入
            for j_id, j_info in self.junctions.items():
                if not j_info['in_links'] or not j_info['out_links']:
                    continue
                if j_info['type'] == 'standard' or len(j_info['out_links']) == 1:
                    for w in j_info['in_links']:
                        out_l = j_info['out_links'][0]
                        Q[self.link_map[out_l]] += P[self.link_map[w]]
                elif j_info['type'] == 'split':
                    tau = self._get_tau(j_info, current_delta[j_id])
                    for (w, z), ratio in tau.items():
                        Q[self.link_map[z]] += ratio * P[self.link_map[w]]
                        
            # 3. 状态推演
            x_next = x_pred + Q - P + demand_forecast[t]
            x_next = np.maximum(0.0, x_next)
            
            # 动态软惩罚 (Soft Penalty)
            for l_id, l_info in self.links.items():
                idx = self.link_map[l_id]
                cap = l_info['capacity']
                if x_next[idx] > cap:
                    violation = x_next[idx] - cap
                    # 降低试错的死亡惩罚，让 GA 敢于尝试把人流往城里送，而不是直接摆烂
                    fitness += 10.0 * (violation ** 2)
                
            # 累加正常的目标函数代价 (Occupancy Penalty + Control Penalty)
            fitness += np.sum(self.weight_Q * (x_next ** 2))
            fitness += self.weight_R * np.sum(individual_u[t] ** 2)
            
            x_pred = x_next
            
        return fitness

    def optimize(self, x_current, demand_forecast):
        # 1. 种群初始化
        pop_delta = np.random.randint(0, self.max_choice, size=(self.pop_size, self.Np, self.num_split))
        pop_u = np.random.randint(0, 2, size=(self.pop_size, self.Np, self.num_nodes))
        
        # 【改进三：启发式初始化与精英移位 (Heuristic Initialization)】
        if self.prev_best_delta is not None:
            # 注入上个时间步的最优解 (将时间轴前移一步)
            shifted_delta = np.roll(self.prev_best_delta, shift=-1, axis=0)
            shifted_u = np.roll(self.prev_best_u, shift=-1, axis=0)
            shifted_delta[-1] = shifted_delta[-2] # 最后一步复制
            shifted_u[-1] = shifted_u[-2]
            
            pop_delta[0] = shifted_delta
            pop_u[0] = shifted_u
            
            # 注入一个静态安全策略 (比如全给 1 作为 Baseline 种子)
            pop_delta[1] = 1 
            pop_u[1] = 1
            
        # 2. 进化主循环
        mutation_rate = 0.05
        
        for iteration in range(self.max_iter):
            fitness_scores = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                fitness_scores[i] = self._evaluate_fitness(pop_delta[i], pop_u[i], x_current, demand_forecast)
                
            # 获取当前代的最优个体 (精英保留 Elitism)
            best_idx = np.argmin(fitness_scores)
            elite_delta = np.copy(pop_delta[best_idx])
            elite_u = np.copy(pop_u[best_idx])
            
            # 选择 (锦标赛选择 Tournament Selection)
            new_pop_delta = np.zeros_like(pop_delta)
            new_pop_u = np.zeros_like(pop_u)
            
            for i in range(self.pop_size):
                # 随机抽取3个，选最好的作为父母
                competitors = np.random.choice(self.pop_size, 3, replace=False)
                winner_idx = competitors[np.argmin(fitness_scores[competitors])]
                new_pop_delta[i] = pop_delta[winner_idx]
                new_pop_u[i] = pop_u[winner_idx]
                
            # 交叉与变异 (Crossover & Mutation)
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    # 均匀交叉
                    mask_delta = np.random.rand(self.Np, self.num_split) > 0.5
                    mask_u = np.random.rand(self.Np, self.num_nodes) > 0.5
                    
                    temp_delta = np.copy(new_pop_delta[i])
                    new_pop_delta[i][mask_delta] = new_pop_delta[i+1][mask_delta]
                    new_pop_delta[i+1][mask_delta] = temp_delta[mask_delta]
                    
                    temp_u = np.copy(new_pop_u[i])
                    new_pop_u[i][mask_u] = new_pop_u[i+1][mask_u]
                    new_pop_u[i+1][mask_u] = temp_u[mask_u]
                    
            # 变异
            mut_mask_delta = np.random.rand(self.pop_size, self.Np, self.num_split) < mutation_rate
            mut_mask_u = np.random.rand(self.pop_size, self.Np, self.num_nodes) < mutation_rate
            
            new_pop_delta[mut_mask_delta] = np.random.randint(0, self.max_choice, size=np.sum(mut_mask_delta))
            new_pop_u[mut_mask_u] = np.random.randint(0, 2, size=np.sum(mut_mask_u))
            
            # 把最优精英强行放回下一代，保证种群绝对不退化
            new_pop_delta[0] = elite_delta
            new_pop_u[0] = elite_u
            
            pop_delta = new_pop_delta
            pop_u = new_pop_u

        # 3. 输出最优解
        # 最后再评估一次最终种群
        final_scores = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            final_scores[i] = self._evaluate_fitness(pop_delta[i], pop_u[i], x_current, demand_forecast)
            
        best_idx = np.argmin(final_scores)
        self.prev_best_delta = np.copy(pop_delta[best_idx])
        self.prev_best_u = np.copy(pop_u[best_idx])
        
        # 取消 Default Fallback
        best_u = {j_id: self.prev_best_u[0, i] for i, j_id in enumerate(self.junctions.keys())}
        
        # 在输出给物理环境前，进行合法指令的取模映射
        best_delta = {}
        for i, j_id in enumerate(self.split_junctions):
            raw_choice = self.prev_best_delta[0, i]
            num_out = len(self.junctions[j_id]['out_links'])
            # 将原始的庞大基因值，映射回该节点合法的 [0, num_out] 指令区间
            best_delta[j_id] = raw_choice % (num_out + 1)
            
        return best_u, best_delta