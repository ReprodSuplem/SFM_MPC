import numpy as np
import matplotlib.pyplot as plt
import random

from sfm_simulator import SFMEnvironment
from mpc_controller import GurobiMPCController
from greedy_controller import GreedyController
from ga_controller import GAController

# ==========================================
# 0. 保证顶刊绝对可复现的全局随机种子
# ==========================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("=== 初始化 SOTA Benchmark: Sioux Falls Network ===")

# ==========================================
# 1. 动态构建 Sioux Falls 拓扑 (24 Nodes, 76 Links)
# ==========================================
# 定义有向边 (start, end, capacity, max_outflow)
# 我们将核心商业区(节点10, 15, 11附近)容量设低(500)，外围环线设高(1200)
sf_edges = [
    (1,2,1200,15), (1,3,1200,15), (2,1,1200,15), (2,6,1200,15), (3,1,1200,15), (3,4,800,10), (3,12,800,10), 
    (4,3,800,10), (4,5,800,10), (4,11,500,8), (5,4,800,10), (5,6,800,10), (5,9,800,10), (6,2,1200,15), 
    (6,5,800,10), (6,8,800,10), (7,8,800,10), (7,18,1200,15), (8,6,800,10), (8,7,800,10), (8,9,500,8), 
    (8,16,500,8), (9,5,800,10), (9,8,500,8), (9,10,500,8), (10,9,500,8), (10,11,500,8), (10,15,500,8), 
    (10,16,500,8), (11,4,500,8), (11,10,500,8), (11,12,800,10), (11,14,500,8), (12,3,800,10), (12,11,800,10), 
    (12,13,1200,15), (13,12,1200,15), (13,24,1200,15), (14,11,500,8), (14,15,500,8), (14,23,800,10), 
    (15,10,500,8), (15,14,500,8), (15,19,800,10), (15,22,500,8), (16,8,500,8), (16,10,500,8), (16,17,800,10), 
    (16,18,800,10), (17,16,800,10), (17,19,800,10), (18,7,1200,15), (18,16,800,10), (18,20,1200,15), 
    (19,15,800,10), (19,17,800,10), (19,20,800,10), (20,18,1200,15), (20,19,800,10), (20,21,1200,15), 
    (20,22,800,10), (21,20,1200,15), (21,22,800,10), (21,24,1200,15), (22,15,500,8), (22,20,800,10), 
    (22,21,800,10), (22,23,800,10), (23,14,800,10), (23,22,800,10), (23,24,1200,15), (24,13,1200,15), 
    (24,21,1200,15), (24,23,1200,15)
]

links = {}
# 添加入口与出口虚拟边
links['L_entry'] = {'capacity': 1e9, 'max_outflow': 50.0, 'start_node': 'Origin', 'end_node': 'N1'}
links['L_exit1'] = {'capacity': 1e9, 'max_outflow': 50.0, 'start_node': 'N20', 'end_node': 'Destination'}
links['L_exit2'] = {'capacity': 1e9, 'max_outflow': 50.0, 'start_node': 'N24', 'end_node': 'Destination'}

# 自动装载 76 条物理路段
for u, v, cap, outf in sf_edges:
    links[f'L{u}_{v}'] = {'capacity': cap, 'max_outflow': outf, 'start_node': f'N{u}', 'end_node': f'N{v}'}

# 动态构建 24 个枢纽节点
junctions = {
    'Origin': {'type': 'standard', 'in_links': [], 'out_links': ['L_entry']},
    'Destination': {'type': 'standard', 'in_links': ['L_exit1', 'L_exit2'], 'out_links': []}
}

for idx in range(1, 25):
    node_name = f'N{idx}'
    in_l = [l_id for l_id, l_info in links.items() if l_info['end_node'] == node_name]
    out_l = [l_id for l_id, l_info in links.items() if l_info['start_node'] == node_name]
    j_type = 'split' if len(out_l) > 1 else 'standard'
    junctions[node_name] = {'type': j_type, 'in_links': in_l, 'out_links': out_l}

num_links = len(links)
link_names = list(links.keys())

# ==========================================
# 2. 仿真全局参数与 Sioux Falls 双峰潮汐客流
# ==========================================
T = 60           
N_steps = 90     
Np = 6           # 考虑到网络庞大，Np=6 保证计算效率

demand_profile = np.zeros((N_steps + Np, num_links))
demand_profile[:, 0] = 300.0                   # 1. 基础负荷 (贯穿全天)
demand_profile[5:25, 0] += 1500.0              # 2. 早高峰 (亚饱和冲击)
demand_profile[45:70, 0] += 2200.0             # 3. 晚高峰 (绝对超饱和冲击，逼迫大范围绕行)

# ==========================================
# 3. 核心仿真函数
# ==========================================
def run_simulation(controller_name):
    print(f"\n[Sioux Falls Benchmark] --- 运行 {controller_name} ---")
    env = SFMEnvironment(links, junctions, T=T, noise_std=0.10)
    
    if controller_name == "MPC":
        controller = GurobiMPCController(links, junctions, T=T, Np=Np)
    elif controller_name == "Greedy":
        controller = GreedyController(links, junctions, threshold=0.80)
    elif controller_name == "GA":
        controller = GAController(links, junctions, T=T, Np=Np, max_iter=20)
        
    x_history = np.zeros((N_steps, num_links))
    x_current = np.zeros(num_links)
    
    for k in range(N_steps):
        demand_forecast = demand_profile[k : k + Np]
        u, delta = controller.optimize(x_current, demand_forecast)
        x_next, Q, P = env.step(x_current, u, delta, demand_profile[k])
        x_history[k] = x_next
        x_current = x_next
        
        if k % 10 == 0:
            sys_total = np.sum(x_current[3:]) # 排除虚拟出入口，计算真实路网总滞留人数
            print(f"Step {k:2d}: Sioux Falls 全网积压人数 = {sys_total:.0f}")
            
    return x_history

# ==========================================
# 4. 执行对比 (这一步极其震撼)
# ==========================================
history_greedy = run_simulation("Greedy")
history_ga = run_simulation("GA")
history_mpc = run_simulation("MPC")

# ==========================================
# 5. Sioux Falls 数据分析与可视化
# ==========================================
# 在复杂网络中，局部路段图已无意义，我们直接看“系统宏观健康度”

def calc_system_occupancy(history):
    # 提取真实物理边 (剔除 L_entry, L_exit1, L_exit2)
    real_idx = [i for i, name in enumerate(link_names) if 'entry' not in name and 'exit' not in name]
    real_caps = np.array([links[link_names[i]]['capacity'] for i in real_idx])
    
    # 1. 计算系统平均拥堵率 (Total Occupancy)
    occ = history[:, real_idx] / real_caps
    avg_occ = np.mean(occ, axis=1)
    
    # 2. 计算全局不平衡度 (Sigma)
    mean_occ_matrix = np.mean(occ, axis=1, keepdims=True)
    sigma = np.sqrt(np.mean((occ - mean_occ_matrix)**2, axis=1))
    
    return avg_occ, sigma

avg_occ_greedy, sigma_greedy = calc_system_occupancy(history_greedy)
avg_occ_ga, sigma_ga = calc_system_occupancy(history_ga)
avg_occ_mpc, sigma_mpc = calc_system_occupancy(history_mpc)

plt.figure(figsize=(14, 5))

# 现象图1：Sioux Falls 全网平均拥堵率 (是否发生了全网大瘫痪？)
plt.subplot(1, 2, 1)
plt.plot(avg_occ_greedy, 'r-.', linewidth=2, label='Greedy')
plt.plot(avg_occ_ga, 'g--', linewidth=1.5, label='GA')
plt.plot(avg_occ_mpc, 'b-', linewidth=2.5, label='MPC')
plt.axhline(0.8, color='gray', linestyle=':', label='High-Risk Threshold')
plt.title("Sioux Falls: System Average Occupancy")
plt.xlabel("Time Step (min)")
plt.ylabel("Average Occupancy Rate")
plt.legend()
plt.grid(True)

# 现象图2：Sioux Falls 全局网络负载均衡度 (76条路段的博弈)
plt.subplot(1, 2, 2)
plt.plot(sigma_greedy, 'r-.', linewidth=2, label='Greedy')
plt.plot(sigma_ga, 'g--', linewidth=1.5, label='GA')
plt.plot(sigma_mpc, 'b-', linewidth=2.5, label='MPC')
plt.title("Sioux Falls: Global Unbalance Index $\sigma(k)$")
plt.xlabel("Time Step (min)")
plt.ylabel("Unbalance $\sigma(k)$ (Lower is Better)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Benchmark_Sioux_Falls.png", dpi=300)
plt.close()

print("\n图表已保存为 Benchmark_Sioux_Falls.png ！")