import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns

def P(sigma, mu):
    return norm.cdf(mu/sigma)

def c(sigma, mu):
    return P(sigma, mu)+0.5*norm.pdf(mu/sigma)/sigma

def pred_CE(sigma, mu):
    c_ = c(sigma, mu)
    Pv = P(sigma, mu)
    pa = norm.pdf(mu/sigma)
    mlp1 = np.log(np.sqrt(2*np.pi)*sigma*c_) + 0.5*((mu/sigma)**2)
    CE_bias = (Pv*(((sigma**2) - (mu**2))/(sigma**2)/2 + mu) + pa*(sigma - 0.5*mu/sigma))/c_
    return mlp1, CE_bias+mlp1

# 原始数据
# 原始数据 (wikimedia/wikipedia, EleutherAI/pythia 全系列)
mus = [
    1.1892951337701507, 
    0.10266660299708083, 
    -1.2604185365046692, 
    -3.488575704144999, 
    -6.417446426765202, 
    -7.9457997506813225, 
    -9.322913564909697, 
    -11.34447103300429, 
    -12.84749016069561, 
    -14.288500393766107
]

sigmas = [
    3.689989500615104, 
    3.855878957136528, 
    4.053293929757489, 
    4.428761293642105, 
    4.821319221881712, 
    4.982977934980508, 
    5.177464976645491, 
    5.4377742973007654, 
    5.585003711322269, 
    5.755553345395172
]

Ss = [
    1189888,        # 14M
    4739072,        # 31M
    18915328,       # 70M
    85056000,       # 160M
    302311424,      # 410M
    805736448,      # 1B
    1208602624,     # 1.4B
    2517652480,     # 2.8B
    6444163072,     # 6.9B
    11327027200     # 12B
]


p_ces, p_mlps = [], []
for mu, sigma in zip(mus, sigmas):
    p_mlp, p_ce = pred_CE(sigma, mu)
    p_ces.append(p_ce)
    p_mlps.append(p_mlp)

S_vals = np.array(Ss)
ce_loss_vals = np.array(p_ces)   # Cross-entropy 预测
mlp_vals    = np.array(p_mlps)   # -log(p1) 预测

# 对数坐标下做线性拟合
logS = np.log10(S_vals)
log_ce = np.log10(ce_loss_vals + 1e-12)
log_mlp = np.log10(mlp_vals + 1e-12)

ce_model = LinearRegression().fit(logS.reshape(-1, 1), log_ce)
mlp_model = LinearRegression().fit(logS.reshape(-1, 1), log_mlp)

ce_pred_log = ce_model.predict(logS.reshape(-1, 1))
mlp_pred_log = mlp_model.predict(logS.reshape(-1, 1))

ce_slope, ce_intercept = ce_model.coef_[0], ce_model.intercept_
mlp_slope, mlp_intercept = mlp_model.coef_[0], mlp_model.intercept_

ce_r2 = r2_score(log_ce, ce_pred_log)
mlp_r2 = r2_score(log_mlp, mlp_pred_log)

# ---------------- 绘图 ----------------
fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()

ax2.set_ylim(0.8, 1.9)

# 配色
colors = sns.color_palette("viridis", 5)
ce_color = colors[3]   # 蓝色
mlp_color = colors[0]  # 绿黄色

# 原始点和线
l1, = ax1.plot(S_vals, ce_loss_vals, marker='o', markersize=4, linewidth=1.8,
               color=ce_color, alpha=0.9, label="Predicted cross-entropy loss")
l2, = ax2.plot(S_vals, mlp_vals, marker='s', markersize=4, linewidth=1.8,
               color=mlp_color, alpha=0.9, label="Predicted $-\log(\\text{RBP}_1)$")

# 拟合线（灰色虚线/点线）
l3, = ax1.plot(S_vals, 10**ce_pred_log, linestyle='--', linewidth=2,
               color="gray", alpha=0.8, label=f"ce fit: slope={ce_slope:.3f}, R²={ce_r2:.3f}")
l4, = ax2.plot(S_vals, 10**mlp_pred_log, linestyle=':', linewidth=2,
               color="gray", alpha=0.8, label="$-\log(\\text{RBP}_1)$" + f" fit: slope={mlp_slope:.3f}, R²={mlp_r2:.3f}")

# 坐标轴设置
ax1.set_xscale("log")
ax1.set_yscale("log")
ax2.set_yscale("log")

ax1.set_xlabel("Model Size S", fontsize=14)
ax1.set_ylabel("cross-entropy loss", fontsize=14, color="black")
ax2.set_ylabel("$-\log(\\text{RBP}_1)$", fontsize=14, color="black")

ax1.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='y', labelcolor="black")

# 合并图例
lines = [l1, l2, l3, l4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, fontsize=9, loc='upper right', frameon=True,
           facecolor='white', edgecolor='gray', framealpha=0.9)

ax1.set_title("Simulated Scaling (topk=1)", fontsize=14, pad=10)

plt.tight_layout()
plt.savefig("CE_vs_logp1_scaling.pdf", dpi=300, bbox_inches="tight")
plt.show()
