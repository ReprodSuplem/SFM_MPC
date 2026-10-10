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
# 1. 定义 Case Study 1 的拓扑结构 (Two-Path Bottleneck)
# ==========================================
# max_outflow: 假设单位为人/秒. 步长 T=60s. 
# 则 T * max_outflow 就是每分钟最大流出人数.
links = {
    'L_entry': {'capacity': 1e9, 'max_outflow': 15.0, 'start_node': 'Origin',  'end_node': 'J_Split'},
    'L_short': {'capacity': 400,  'max_outflow': 4.0,  'start_node': 'J_Split', 'end_node': 'J_Merge'}, # 瓶颈1
    'L_long':  {'capacity': 800,  'max_outflow': 8.0,  'start_node': 'J_Split', 'end_node': 'J_Merge'}, # 瓶颈2
    'L_exit':  {'capacity': 1e9, 'max_outflow': 20.0, 'start_node': 'J_Merge', 'end_node': 'Destination'}
}

junctions = {
    'Origin':      {'type': 'standard', 'in_links': [], 'out_links': ['L_entry']},
    'J_Split':     {'type': 'split',    'in_links': ['L_entry'], 'out_links': ['L_short', 'L_long']},
    'J_Merge':     {'type': 'standard', 'in_links': ['L_short', 'L_long'], 'out_links': ['L_exit']},
    'Destination': {'type': 'standard', 'in_links': ['L_exit'], 'out_links': []}
}

# ==========================================
# 2. 仿真全局参数与需求注入 (Demand Surge)
# ==========================================
T = 60           # 控制周期 60秒 (1分钟)
N_steps = 90     # 总仿真时长 90 分钟
Np = 5           # MPC 预测时域 5 分钟

num_links = len(links)
link_names = list(links.keys())
idx_short = link_names.index('L_short')
idx_long = link_names.index('L_long')

demand_profile = np.zeros((N_steps + Np, num_links))

# 1. 基础人流
demand_profile[:, 0] = 50.0 
# 2. 第一波高峰 (早高峰): 总计 550 人/分 (在系统处理能力 720 之内，测试正常分流)
demand_profile[5:25, 0] += 500.0  
# 3. 第二波高峰 (晚高峰): 总计 750 人/分 (击穿 720 的理论极限，测试极限抗压与限流)
demand_profile[45:70, 0] += 700.0

# ==========================================
# 3. 运行仿真的辅助函数
# ==========================================
def run_simulation(controller_name):
    print(f"\n--- 正在运行 {controller_name} ---")
    
    # 实例化带有 10% 客观物理噪声的环境 (测试鲁棒性)
    env = SFMEnvironment(links, junctions, T=T, noise_std=0.10)
    
    # 实例化控制器
    if controller_name == "MPC":
        controller = GurobiMPCController(links, junctions, T=T, Np=Np)
    elif controller_name == "Greedy":
        controller = GreedyController(links, junctions, threshold=0.85)
    elif controller_name == "GA":
        controller = GAController(links, junctions, T=T, Np=Np, max_iter=20)
        
    x_history = np.zeros((N_steps, num_links))
    x_current = np.zeros(num_links)
    
    for k in range(N_steps):
        # 提取给控制器的未来需求预测 (假设预测是无噪声的准确均值)
        demand_forecast = demand_profile[k : k + Np]
        
        # 控制器计算最优指令
        u, delta = controller.optimize(x_current, demand_forecast)
        
        # 将指令下发给物理环境执行
        x_next, Q, P = env.step(x_current, u, delta, demand_profile[k])
        
        x_history[k] = x_next
        x_current = x_next
        
        # 打印部分日志观察进展
        if k % 5 == 0:
            print(f"Step {k}: 短路径={x_current[idx_short]:.1f}/400, 长路径={x_current[idx_long]:.1f}/800, delta={delta.get('J_Split',0)}")
            
    return x_history

# ==========================================
# 4. 执行三大控制器对比
# ==========================================
history_greedy = run_simulation("Greedy")
history_ga = run_simulation("GA")
history_mpc = run_simulation("MPC")

# ==========================================
# 5. 绘制对比图 (供论文使用)
# ==========================================
# 计算占据率 (Occupancy = x / Capacity)
cap_short = links['L_short']['capacity']
cap_long = links['L_long']['capacity']

occ_short_greedy = history_greedy[:, idx_short] / cap_short
occ_long_greedy = history_greedy[:, idx_long] / cap_long

occ_short_ga = history_ga[:, idx_short] / cap_short
occ_long_ga = history_ga[:, idx_long] / cap_long

occ_short_mpc = history_mpc[:, idx_short] / cap_short
occ_long_mpc = history_mpc[:, idx_long] / cap_long

plt.figure(figsize=(18, 5))

# 子图1：贪心算法 (Ping-pong 效应)
plt.subplot(1, 3, 1)
plt.plot(occ_short_greedy, 'r--', label='Short Path (Greedy)')
plt.plot(occ_long_greedy, 'b--', label='Long Path (Greedy)')
plt.axhline(0.85, color='gray', linestyle=':', label='Threshold (85%)')
plt.title("Greedy: Ping-pong Effect")
plt.xlabel("Time Step (min)")
plt.ylabel("Occupancy Rate")
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True)

# 子图2：遗传算法 (无序震荡)
plt.subplot(1, 3, 2)
plt.plot(occ_short_ga, 'r--', label='Short Path (GA)')
plt.plot(occ_long_ga, 'b--', label='Long Path (GA)')
plt.axhline(1.0, color='gray', linestyle=':', label='Max Capacity (100%)')
plt.title("GA: Sub-optimal Oscillation")
plt.xlabel("Time Step (min)")
plt.ylabel("Occupancy Rate")
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True)

# 子图3：MPC (平滑负载均衡)
plt.subplot(1, 3, 3)
plt.plot(occ_short_mpc, 'r-', label='Short Path (MPC)')
plt.plot(occ_long_mpc, 'b-', label='Long Path (MPC)')
plt.axhline(1.0, color='gray', linestyle=':', label='Max Capacity (100%)')
plt.title("Gurobi MPC: Smooth Load Balancing")
plt.xlabel("Time Step (min)")
plt.ylabel("Occupancy Rate")
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Case1_Occupancy_Comparison.png", dpi=300)
plt.close()

# ==========================================
# 6. 计算分布均衡度 sigma(k) 对应原论文 Eq. 5.2
# ==========================================
def calc_sigma(history):
    # 仅针对并行的两条路进行负载均衡计算
    occ = history[:, [idx_short, idx_long]] / np.array([cap_short, cap_long])
    mean_occ = np.mean(occ, axis=1, keepdims=True)
    sigma = np.sqrt(np.mean((occ - mean_occ)**2, axis=1))
    return sigma

sigma_greedy = calc_sigma(history_greedy)
sigma_ga = calc_sigma(history_ga)
sigma_mpc = calc_sigma(history_mpc)

plt.figure(figsize=(8, 5))
plt.plot(sigma_greedy, 'k-.', label='Greedy (Reactive)')
plt.plot(sigma_ga, 'g--', label='GA (Heuristic)')
plt.plot(sigma_mpc, 'b-', linewidth=2, label='MPC (Proposed)')
plt.title("Crowd Distribution Balance $\sigma(k)$ over Time")
plt.xlabel("Time Step (min)")
plt.ylabel("Unbalance Index $\sigma(k)$ (Lower is better)")
plt.legend()
plt.grid(True)
plt.savefig("Case1_Sigma_Comparison.png", dpi=300)
plt.close()