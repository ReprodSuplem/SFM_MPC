import numpy as np
import matplotlib.pyplot as plt
import random

from sfm_simulator import SFMEnvironment
from mpc_controller import GurobiMPCController

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)  # 固定 Numpy 的随机序列 (控制 GA 的变异和环境的高斯噪声)
random.seed(RANDOM_SEED)     # 固定 Python 内置库的随机序列

# 沿用 Case 3 曼哈顿网格的极简拓扑配置 (代码结构保持一致)
links = {
    'L_entry': {'capacity': 1e9, 'max_outflow': 30.0, 'start_node': 'Origin', 'end_node': 'N00'},
    'L00_01': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N00', 'end_node': 'N01'},
    'L01_02': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N01', 'end_node': 'N02'},
    'L10_11': {'capacity': 600, 'max_outflow': 8.0,  'start_node': 'N10', 'end_node': 'N11'}, 
    'L11_12': {'capacity': 300, 'max_outflow': 4.0,  'start_node': 'N11', 'end_node': 'N12'}, # 中心瓶颈
    'L20_21': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N20', 'end_node': 'N21'},
    'L21_22': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N21', 'end_node': 'N22'},
    'L00_10': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N00', 'end_node': 'N10'},
    'L01_11': {'capacity': 600, 'max_outflow': 8.0,  'start_node': 'N01', 'end_node': 'N11'}, 
    'L02_12': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N02', 'end_node': 'N12'},
    'L10_20': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N10', 'end_node': 'N20'},
    'L11_21': {'capacity': 300, 'max_outflow': 4.0,  'start_node': 'N11', 'end_node': 'N21'}, # 中心瓶颈
    'L12_22': {'capacity': 800, 'max_outflow': 10.0, 'start_node': 'N12', 'end_node': 'N22'},
    'L_exit': {'capacity': 1e9, 'max_outflow': 40.0, 'start_node': 'N22', 'end_node': 'Destination'}
}

junctions = {
    'Origin': {'type': 'standard', 'in_links': [], 'out_links': ['L_entry']},
    'N00': {'type': 'split', 'in_links': ['L_entry'], 'out_links': ['L00_01', 'L00_10']},
    'N01': {'type': 'split', 'in_links': ['L00_01'],  'out_links': ['L01_02', 'L01_11']},
    'N10': {'type': 'split', 'in_links': ['L00_10'],  'out_links': ['L10_11', 'L10_20']},
    'N11': {'type': 'split', 'in_links': ['L10_11', 'L01_11'], 'out_links': ['L11_12', 'L11_21']},
    'N02': {'type': 'standard', 'in_links': ['L01_02'], 'out_links': ['L02_12']},
    'N20': {'type': 'standard', 'in_links': ['L10_20'], 'out_links': ['L20_21']},
    'N12': {'type': 'standard', 'in_links': ['L02_12', 'L11_12'], 'out_links': ['L12_22']},
    'N21': {'type': 'standard', 'in_links': ['L11_21', 'L20_21'], 'out_links': ['L21_22']},
    'N22': {'type': 'standard', 'in_links': ['L12_22', 'L21_22'], 'out_links': ['L_exit']},
    'Destination': {'type': 'standard', 'in_links': ['L_exit'], 'out_links': []}
}

T = 60           
N_steps = 90
Np = 6           

num_links = len(links)
link_names = list(links.keys())

# ==========================================
# 构建完全匹配 Case 3 的双峰潮汐客流
# ==========================================
demand_profile = np.zeros((N_steps + Np, num_links))

# 1. 基础人流 
demand_profile[:, 0] = 150.0 
# 2. 第一波高峰: 追加 800 人/分 (总计 950)
demand_profile[5:25, 0] += 800.0  
# 3. 第二波超大高峰: 追加 1000 人/分 (总计 1150)
demand_profile[45:70, 0] += 1000.0  

# 待测试的噪声水平序列
noise_levels = [0.0, 0.10, 0.20, 0.30]
results_history = {}

print("=== 开始鲁棒性边界测试 (Robustness Bound Test) ===")

for noise in noise_levels:
    print(f"\n>> 正在测试环境噪声强度: {noise*100}%")
    env = SFMEnvironment(links, junctions, T=T, noise_std=noise)
    controller = GurobiMPCController(links, junctions, T=T, Np=Np)
    
    x_history = np.zeros((N_steps, num_links))
    x_current = np.zeros(num_links)
    
    for k in range(N_steps):
        demand_forecast = demand_profile[k : k + Np]
        u, delta = controller.optimize(x_current, demand_forecast)
        x_next, Q, P = env.step(x_current, u, delta, demand_profile[k])
        x_history[k] = x_next
        x_current = x_next
        
    results_history[noise] = x_history

# ==========================================
# 绘制鲁棒性边界分析图
# ==========================================
internal_links = ['L00_01', 'L01_02', 'L10_11', 'L11_12', 'L20_21', 'L21_22',
                  'L00_10', 'L01_11', 'L02_12', 'L10_20', 'L11_21', 'L12_22']
idx_internal = [link_names.index(name) for name in internal_links]
cap_internal = np.array([links[name]['capacity'] for name in internal_links])

def calc_sigma_grid(history):
    occ = history[:, idx_internal] / cap_internal
    mean_occ = np.mean(occ, axis=1, keepdims=True)
    sigma = np.sqrt(np.mean((occ - mean_occ)**2, axis=1))
    return sigma

plt.figure(figsize=(10, 6))

colors = ['b', 'g', 'orange', 'r']
styles = ['-', '--', '-.', ':']

for i, noise in enumerate(noise_levels):
    sigma = calc_sigma_grid(results_history[noise])
    plt.plot(sigma, color=colors[i], linestyle=styles[i], linewidth=2.5, 
             label=f'MPC under {int(noise*100)}% Parameter Mismatch')

plt.title("Robustness Bound: $\sigma(k)$ Degradation under Environmental Noise")
plt.xlabel("Time Step (min)")
plt.ylabel("Global Unbalance Index $\sigma(k)$")
plt.ylim(0, 0.4)
plt.axhline(0.15, color='gray', linestyle='-', alpha=0.5, label='Safety Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Robustness_Analysis.png", dpi=300)
plt.close()

print("\n图表已保存为 Robustness_Analysis.png ！")