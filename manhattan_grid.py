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
# 1. 定义 Case 3 拓扑: 3x3 曼哈顿网格 (Manhattan Grid)
# ==========================================
# 坐标系: N00(左上角, 起点) -> N22(右下角, 终点)
# 流量只能向右(East)或向下(South)移动。
# 我们故意在中心区域 (N11周围) 设置极窄的容量瓶颈，
# 促使算法必须提早在 N00, N01, N10 就进行向外围的绕行分流。

links = {
    'L_entry': {'capacity': 1e9, 'max_outflow': 30.0, 'start_node': 'Origin', 'end_node': 'N00'},
    
    # --- 横向路段 (Right) ---
    'L00_01': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N00', 'end_node': 'N01'},
    'L01_02': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N01', 'end_node': 'N02'},
    'L10_11': {'capacity': 600, 'max_outflow': 8.0,  'start_node': 'N10', 'end_node': 'N11'}, # 趋近中心
    'L11_12': {'capacity': 300, 'max_outflow': 4.0,  'start_node': 'N11', 'end_node': 'N12'}, # 中心瓶颈！
    'L20_21': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N20', 'end_node': 'N21'},
    'L21_22': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N21', 'end_node': 'N22'},

    # --- 纵向路段 (Down) ---
    'L00_10': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N00', 'end_node': 'N10'},
    'L01_11': {'capacity': 600, 'max_outflow': 8.0,  'start_node': 'N01', 'end_node': 'N11'}, # 趋近中心
    'L02_12': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N02', 'end_node': 'N12'},
    'L10_20': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N10', 'end_node': 'N20'},
    'L11_21': {'capacity': 300, 'max_outflow': 4.0,  'start_node': 'N11', 'end_node': 'N21'}, # 中心瓶颈！
    'L12_22': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N12', 'end_node': 'N22'},
    
    'L_exit': {'capacity': 1e9, 'max_outflow': 40.0, 'start_node': 'N22', 'end_node': 'Destination'}
}

# out_links 的顺序统一为: [0] 是向右(Right), [1] 是向下(Down)
junctions = {
    'Origin': {'type': 'standard', 'in_links': [], 'out_links': ['L_entry']},
    
    # 4 个分流节点 (Split)
    'N00': {'type': 'split', 'in_links': ['L_entry'], 'out_links': ['L00_01', 'L00_10']},
    'N01': {'type': 'split', 'in_links': ['L00_01'],  'out_links': ['L01_02', 'L01_11']},
    'N10': {'type': 'split', 'in_links': ['L00_10'],  'out_links': ['L10_11', 'L10_20']},
    'N11': {'type': 'split', 'in_links': ['L10_11', 'L01_11'], 'out_links': ['L11_12', 'L11_21']}, # 中心枢纽
    
    # 5 个标准汇合/转角节点 (Standard)
    'N02': {'type': 'standard', 'in_links': ['L01_02'], 'out_links': ['L02_12']},
    'N20': {'type': 'standard', 'in_links': ['L10_20'], 'out_links': ['L20_21']},
    'N12': {'type': 'standard', 'in_links': ['L02_12', 'L11_12'], 'out_links': ['L12_22']},
    'N21': {'type': 'standard', 'in_links': ['L11_21', 'L20_21'], 'out_links': ['L21_22']},
    'N22': {'type': 'standard', 'in_links': ['L12_22', 'L21_22'], 'out_links': ['L_exit']},
    
    'Destination': {'type': 'standard', 'in_links': ['L_exit'], 'out_links': []}
}

# ==========================================
# 2. 仿真全局参数与网格高压需求 (双峰潮汐客流)
# ==========================================
T = 60           
N_steps = 90     # 网格大，人流清空需要更长时间，设为 90 分钟
Np = 6           # 加大预测时域以看清全局，设为 6 分钟

num_links = len(links)
link_names = list(links.keys())

demand_profile = np.zeros((N_steps + Np, num_links))

# 1. 基础人流 (Background Demand): 全程存在少量人流，防止图表归 0
demand_profile[:, 0] = 150.0
# 2. 第一波高峰 (Peak 1): 第 5 到 25 分钟，追加 800 人/分 (总计 950人/分)
demand_profile[5:25, 0] += 800.0
# 3. 第二波超大高峰 (Peak 2): 第 45 到 70 分钟，爆发更猛烈的 1000 人/分 (总计 1150人/分)
demand_profile[45:70, 0] += 1000.0

# ==========================================
# 3. 核心仿真函数
# ==========================================
def run_simulation(controller_name):
    print(f"\n[Case 3 3x3 Manhattan Grid] --- 运行 {controller_name} ---")
    
    # 引入 15% 物理噪声，极致考验鲁棒性
    env = SFMEnvironment(links, junctions, T=T, noise_std=0.15)
    
    if controller_name == "MPC":
        controller = GurobiMPCController(links, junctions, T=T, Np=Np)
    elif controller_name == "Greedy":
        controller = GreedyController(links, junctions, threshold=0.75)
    elif controller_name == "GA":
        controller = GAController(links, junctions, T=T, Np=Np, pop_size=60, max_iter=30)
        
    x_history = np.zeros((N_steps, num_links))
    x_current = np.zeros(num_links)
    
    for k in range(N_steps):
        demand_forecast = demand_profile[k : k + Np]
        u, delta = controller.optimize(x_current, demand_forecast)
        x_next, Q, P = env.step(x_current, u, delta, demand_profile[k])
        x_history[k] = x_next
        x_current = x_next
        
        if k % 10 == 0:
            print(f"Step {k:2d}: 中心死穴 L11_12={x_current[link_names.index('L11_12')]:.0f}/300, 外围 L01_02={x_current[link_names.index('L01_02')]:.0f}/800")
            
    return x_history

# ==========================================
# 4. 执行三大控制器
# ==========================================
history_greedy = run_simulation("Greedy")
history_ga = run_simulation("GA")
history_mpc = run_simulation("MPC")

# ==========================================
# 5. 可视化分析 (网格局部拥堵 vs 全局均衡)
# ==========================================
# 我们提取中心区域 (L11_12, L11_21) 和外围区域 (L01_02, L10_20) 做对比
idx_center = [link_names.index('L11_12'), link_names.index('L11_21')]
idx_outer  = [link_names.index('L01_02'), link_names.index('L10_20')]

def get_avg_occupancy(history, indices, capacities):
    occ = history[:, indices] / np.array(capacities)
    return np.mean(occ, axis=1)

cap_center = [links['L11_12']['capacity'], links['L11_21']['capacity']]
cap_outer  = [links['L01_02']['capacity'], links['L10_20']['capacity']]

occ_center_greedy = get_avg_occupancy(history_greedy, idx_center, cap_center)
occ_center_ga     = get_avg_occupancy(history_ga, idx_center, cap_center)
occ_center_mpc    = get_avg_occupancy(history_mpc, idx_center, cap_center)

occ_outer_greedy  = get_avg_occupancy(history_greedy, idx_outer, cap_outer)
occ_outer_ga      = get_avg_occupancy(history_ga, idx_outer, cap_outer)
occ_outer_mpc     = get_avg_occupancy(history_mpc, idx_outer, cap_outer)

plt.figure(figsize=(15, 5))

# 现象图1：网格中心死锁 (Gridlock)
plt.subplot(1, 2, 1)
plt.plot(occ_center_greedy, 'r-.', linewidth=2, label='Center (Greedy)')
plt.plot(occ_center_ga, 'g--', linewidth=1.5, label='Center (GA)')
plt.plot(occ_center_mpc, 'b-', linewidth=2.5, label='Center (MPC)')
plt.axhline(1.0, color='gray', linestyle=':', label='Max Capacity')
plt.title("Gridlock in Center Bottlenecks")
plt.xlabel("Time Step (min)")
plt.ylabel("Average Occupancy Rate")
plt.ylim(0, 1.3)
plt.legend()
plt.grid(True)

# 现象图2：网格边缘绕行利用率
plt.subplot(1, 2, 2)
plt.plot(occ_outer_greedy, 'r-.', linewidth=2, label='Outer Ring (Greedy)')
plt.plot(occ_outer_ga, 'g--', linewidth=1.5, label='Outer Ring (GA)')
plt.plot(occ_outer_mpc, 'b-', linewidth=2.5, label='Outer Ring (MPC)')
plt.title("Utilization of Outer Routes (Bypassing)")
plt.xlabel("Time Step (min)")
plt.ylabel("Average Occupancy Rate")
plt.ylim(0, 1.3)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Case3_Manhattan_Grid_Comparison.png", dpi=300)
plt.close()

# 计算 12 条核心网格内部道路的全局负载均衡
internal_links = ['L00_01', 'L01_02', 'L10_11', 'L11_12', 'L20_21', 'L21_22',
                  'L00_10', 'L01_11', 'L02_12', 'L10_20', 'L11_21', 'L12_22']
idx_internal = [link_names.index(name) for name in internal_links]
cap_internal = np.array([links[name]['capacity'] for name in internal_links])

def calc_sigma_grid(history):
    occ = history[:, idx_internal] / cap_internal
    mean_occ = np.mean(occ, axis=1, keepdims=True)
    sigma = np.sqrt(np.mean((occ - mean_occ)**2, axis=1))
    return sigma

plt.figure(figsize=(8, 5))
plt.plot(calc_sigma_grid(history_greedy), 'r-.', label='Greedy (Gridlock)')
plt.plot(calc_sigma_grid(history_ga), 'g--', label='GA (Curse of Dimensionality)')
plt.plot(calc_sigma_grid(history_mpc), 'b-', linewidth=2.5, label='Gurobi MPC (Scalable)')
plt.title("Global Network Unbalance Index $\sigma(k)$ in 3x3 Grid")
plt.xlabel("Time Step (min)")
plt.ylabel("Unbalance $\sigma(k)$ (Lower is Better)")
plt.legend()
plt.grid(True)
plt.savefig("Case3_Sigma_Comparison.png", dpi=300)
plt.close()

print("\n图表已保存为 Case3_Manhattan_Grid_Comparison.png 和 Case3_Sigma_Comparison.png ！")