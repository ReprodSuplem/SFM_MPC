import numpy as np
import matplotlib.pyplot as plt
import random

from sfm_simulator import SFMEnvironment
from mpc_controller import GurobiMPCController
from greedy_controller import GreedyController
from ga_controller import GAController

# ==========================================
# 0. 全局随机种子设置 (保证论文实验绝对可复现)
# ==========================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)  # 固定 Numpy 的随机序列 (控制 GA 的变异和环境的高斯噪声)
random.seed(RANDOM_SEED)     # 固定 Python 内置库的随机序列

# ==========================================
# 1. 定义 Case 2 拓扑: 经典的布雷斯悖论路网
# ==========================================
# 拓扑结构：
#          /-- L_SA(宽但绕) --\
#  Origin -            (捷径 L_AB)       -- J_End -- Dest
#          \-- L_SB(窄但近) --/
# 
# 逻辑：L_AB 是捷径。如果不加协调，所有人都在 J_Start 走 L_SA，在 J_A 走 L_AB，
# 结果全部挤在 J_B 的出口，导致链式拥堵。
links = {
    'L_entry': {'capacity': 1e9, 'max_outflow': 20.0, 'start_node': 'Origin',  'end_node': 'J_Start'},
    
    # 上路 (SA - AEnd): 前宽后窄
    'L_SA':    {'capacity': 800,  'max_outflow': 10.0, 'start_node': 'J_Start', 'end_node': 'J_A'},
    'L_AEnd':  {'capacity': 400,  'max_outflow': 5.0,  'start_node': 'J_A',     'end_node': 'J_End'},
    
    # 下路 (SB - BEnd): 前窄后宽
    'L_SB':    {'capacity': 400,  'max_outflow': 5.0,  'start_node': 'J_Start', 'end_node': 'J_B'},
    'L_BEnd':  {'capacity': 800,  'max_outflow': 10.0, 'start_node': 'J_B',     'end_node': 'J_End'},
    
    # 诱人的捷径 (Shortcut): 容量极小，但流率高
    'L_AB':    {'capacity': 200,  'max_outflow': 12.0, 'start_node': 'J_A',     'end_node': 'J_B'},
    
    'L_exit':  {'capacity': 1e9, 'max_outflow': 25.0, 'start_node': 'J_End',   'end_node': 'Destination'}
}

junctions = {
    'Origin':      {'type': 'standard', 'in_links': [], 'out_links': ['L_entry']},
    # 起点分流
    'J_Start':     {'type': 'split',    'in_links': ['L_entry'], 'out_links': ['L_SA', 'L_SB']},
    # J_A 节点：可以直行去 End，也可以贪心走捷径 L_AB
    'J_A':         {'type': 'split',    'in_links': ['L_SA'],    'out_links': ['L_AB', 'L_AEnd']},
    
    'J_B':         {'type': 'standard', 'in_links': ['L_SB', 'L_AB'], 'out_links': ['L_BEnd']},
    'J_End':       {'type': 'standard', 'in_links': ['L_AEnd', 'L_BEnd'], 'out_links': ['L_exit']},
    'Destination': {'type': 'standard', 'in_links': ['L_exit'], 'out_links': []}
}

# ==========================================
# 2. 仿真全局参数与大客流注入 (Surge Demand)
# ==========================================
T = 60           # 1分钟步长
N_steps = 90     # 仿真时长 90 分钟
Np = 5           # MPC预测未来5分钟

num_links = len(links)
link_names = list(links.keys())

demand_profile = np.zeros((N_steps + Np, num_links))

# 1. 基础人流
demand_profile[:, 0] = 80.0 
# 2. 第一波高峰: 诱导贪心算法掉入捷径陷阱
demand_profile[5:25, 0] += 500.0  
# 3. 第二波高峰: 全面施压，总计 780 人/分
demand_profile[45:70, 0] += 700.0

# ==========================================
# 3. 核心仿真函数
# ==========================================
def run_simulation(controller_name):
    print(f"\n[Case 2 Braess's Paradox] --- 运行 {controller_name} ---")
    
    # 注入 10% 客观环境噪声
    env = SFMEnvironment(links, junctions, T=T, noise_std=0.10)
    
    if controller_name == "MPC":
        controller = GurobiMPCController(links, junctions, T=T, Np=Np)
    elif controller_name == "Greedy":
        # 阈值设为 80%
        controller = GreedyController(links, junctions, threshold=0.80)
    elif controller_name == "GA":
        controller = GAController(links, junctions, T=T, Np=Np, max_iter=25)
        
    x_history = np.zeros((N_steps, num_links))
    x_current = np.zeros(num_links)
    
    for k in range(N_steps):
        demand_forecast = demand_profile[k : k + Np]
        u, delta = controller.optimize(x_current, demand_forecast)
        x_next, Q, P = env.step(x_current, u, delta, demand_profile[k])
        x_history[k] = x_next
        x_current = x_next
        
        # 观察 J_A 节点的选择 (是否陷入捷径陷阱)
        if k % 5 == 0:
            print(f"Step {k:2d}: 捷径 L_AB 人数={x_current[link_names.index('L_AB')]:.0f}/200, J_A 决策(1:走捷径, 2:走主路, 0:均分) = {delta.get('J_A',0)}")
            
    return x_history

# ==========================================
# 4. 执行三大控制器
# ==========================================
history_greedy = run_simulation("Greedy")
history_ga = run_simulation("GA")
history_mpc = run_simulation("MPC")

# ==========================================
# 5. 可视化分析 (直接生成图表)
# ==========================================
idx_AB = link_names.index('L_AB')     # 诱人的捷径 (最容易爆满瘫痪)
idx_BEnd = link_names.index('L_BEnd') # 捷径的下游 (链式瘫痪受害者)

# 计算核心受灾区 (捷径 + 捷径下游) 的总拥堵率
occ_greedy = (history_greedy[:, idx_AB] + history_greedy[:, idx_BEnd]) / (200 + 800)
occ_ga = (history_ga[:, idx_AB] + history_ga[:, idx_BEnd]) / (200 + 800)
occ_mpc = (history_mpc[:, idx_AB] + history_mpc[:, idx_BEnd]) / (200 + 800)

plt.figure(figsize=(12, 5))

# 现象图1：捷径诱导的局部瘫痪 (Ping-pong and Deadlock)
plt.subplot(1, 2, 1)
plt.plot(occ_greedy, 'r-.', linewidth=2, label='Greedy (Fell into Trap)')
plt.plot(occ_ga, 'g--', linewidth=1.5, label='GA (Sub-optimal)')
plt.plot(occ_mpc, 'b-', linewidth=2.5, label='MPC (Bypassed Trap)')
plt.axhline(1.0, color='gray', linestyle=':', label='Max Capacity')
plt.title("Congestion on Shortcut Route (L_AB & L_BEnd)")
plt.xlabel("Time Step (min)")
plt.ylabel("Average Occupancy Rate")
plt.ylim(0, 1.2)
plt.legend()
plt.grid(True)

# 现象图2：全局网络负载均衡度 (Sigma)
def calc_sigma_all_internal(history):
    # 计算内部 5 条支路的均衡度
    internal_indices = [link_names.index(name) for name in ['L_SA', 'L_AEnd', 'L_SB', 'L_BEnd', 'L_AB']]
    caps = np.array([links[name]['capacity'] for name in ['L_SA', 'L_AEnd', 'L_SB', 'L_BEnd', 'L_AB']])
    
    occ = history[:, internal_indices] / caps
    mean_occ = np.mean(occ, axis=1, keepdims=True)
    sigma = np.sqrt(np.mean((occ - mean_occ)**2, axis=1))
    return sigma

plt.subplot(1, 2, 2)
plt.plot(calc_sigma_all_internal(history_greedy), 'r-.', label='Greedy')
plt.plot(calc_sigma_all_internal(history_ga), 'g--', label='GA')
plt.plot(calc_sigma_all_internal(history_mpc), 'b-', linewidth=2.5, label='MPC')
plt.title("Global Network Unbalance Index $\sigma(k)$")
plt.xlabel("Time Step (min)")
plt.ylabel("Unbalance $\sigma(k)$ (Lower is Better)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Case2_Braess_Paradox_Comparison.png", dpi=300)
plt.close()

print("\n图表已保存为 Case2_Braess_Paradox_Comparison.png ！")