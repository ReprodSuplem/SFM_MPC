import numpy as np

class GreedyController:
    """
    基于阈值的局部贪心反馈控制器 (Greedy Feedback Control)
    完全无预测能力 (Reactive)，模拟传统的人工“看图调度”逻辑。
    """
    def __init__(self, links, junctions, threshold=0.8):
        """
        :param links: 路段属性字典
        :param junctions: 节点属性字典
        :param threshold: 拥堵报警阈值 (默认 80% 占用率)，超过此值即触发信号切换
        """
        self.links = links
        self.junctions = junctions
        self.threshold = threshold
        self.link_map = {l_id: idx for idx, l_id in enumerate(links.keys())}
        
    def optimize(self, x_current, demand_forecast=None):
        """
        根据当前状态 x_current 给出即时反应。
        注：demand_forecast 在此算法中被故意完全忽略，凸显其短视特征。
        """
        best_u = {}
        best_delta = {}
        
        for j_id, j_info in self.junctions.items():
            # 默认所有节点放行 (标准信号全部亮绿灯)
            best_u[j_id] = 1.0  
            
            if j_info['type'] == 'split':
                out_links = j_info['out_links']
                num_out = len(out_links)
                
                if num_out == 0:
                    continue
                    
                # 1. 获取所有下游分支当前的真实占有率 (Occupancy)
                occupancies = []
                for out_l in out_links:
                    idx = self.link_map[out_l]
                    cap = self.links[out_l]['capacity']
                    # 保护机制，防止容量为0的除0错误
                    occ = x_current[idx] / cap if cap > 0 else 1.0
                    occupancies.append(occ)
                    
                # 2. 贪心路由逻辑 (Greedy Routing)
                # 假设 out_links[0] 是首选的主干道（即物理距离最短的路径）
                primary_occ = occupancies[0]
                
                if primary_occ < self.threshold:
                    # 首选路径未拥堵 -> 全部分流给首选路径
                    # 在环境设定中，choice = 1 代表 100% 导向 out_links[0]
                    best_delta[j_id] = 1 
                    
                else:
                    # 首选路径已拥堵 -> 寻找当前最空闲的备用路径
                    best_out_idx = int(np.argmin(occupancies))
                    min_occ = occupancies[best_out_idx]
                    
                    if min_occ > 0.95:
                        # 极端情况：所有备用路径也都几乎爆满 (> 95%)
                        # 此时瞬间全切会引发死锁，退化为均分策略 (50/50分流)
                        # 在环境设定中，choice = 0 代表均分
                        best_delta[j_id] = 0 
                        
                        # 甚至可以在这里加入限流逻辑（掐断上游红绿灯），
                        # 但为了对比公平，我们仅依靠分流，展示 Greedy 的无奈
                        # best_u[j_id] = 0.0 
                    else:
                        # 全力将人流导向当前最空闲的路径
                        # choice 为 index + 1
                        best_delta[j_id] = best_out_idx + 1
                        
        return best_u, best_delta