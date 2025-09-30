import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.interpolate import griddata
import seaborn as sns
from matplotlib.ticker import LogLocator, FuncFormatter
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter  # 可选，用于平滑
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from utils import *

def plot_logCE_vs_logS(dataset_name, model_sizes, topk): # CE scaling law
    metrics = collect_metrics(topk, dataset_name=dataset_name, base_result_dir="eval_gt_in_topk",considered_model_names=pythia_models)
    if "mean_ce_loss" not in metrics:
        print("❌ metrics.json 里没有 mean_ce_loss")
        return None, None

    x = np.log10(model_sizes).reshape(-1, 1)
    y = np.log10(metrics["mean_ce_loss"])
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    alpha_CE = -model.coef_[0]  # 注意这里 logCE = -alpha*logS + C
    C = np.exp(model.intercept_)  # intercept 对应 C

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, color="#1f77b4", s=40, label="CE (data)")  # 深蓝实心圆
    plt.plot(x, y_pred, color="#1f77b4", linestyle="--", alpha=0.6, linewidth=2, label=f"Fit: logCE={-alpha_CE:.4f}*logS+{np.log(C):.4f}")  # 同色虚线
    plt.xlabel("log(Model Size)")
    plt.ylabel("log(Cross-Entropy Loss)")
    plt.title("Scaling Law: log(CE) vs log(Model Size)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'assets/logCE_vs_logS_topk={topk}.png')

    print(f"R² = {r2:.4f}")
    print(f"Fitted line: log(CE) = -{alpha_CE:.4f} * log(S) + log({C:.4f})")
    

def plot_CE_and_rankloss_vs_S(model_sizes, topk=1):
    # 采集数据
    metrics = collect_metrics(topk, base_result_dir="eval_gt_in_topk")

    S_vals = np.array(model_sizes)
    ce_loss_vals = np.array(metrics["mean_ce_loss"])             # CE loss
    indicator_vals = np.array(metrics["mean_indicator"])              # rank-based 指标
    rank_loss_vals = -np.log(indicator_vals + 1e-12)           # -log(indicator)

    logS = np.log10(S_vals)

    fig, ax1 = plt.subplots(figsize=(7,4))

    # 调整配色：选择 viridis 的 0.2 和 0.8，避免太相似
    colors = sns.color_palette("viridis", 5)
    ce_color = colors[3]   # 偏蓝
    rank_color = colors[0] # 偏黄绿

    # ---------------- CE loss 拟合 ----------------
    log_ce = np.log10(ce_loss_vals + 1e-12)
    ce_model = LinearRegression()
    ce_model.fit(logS.reshape(-1, 1), log_ce)
    ce_pred_log = ce_model.predict(logS.reshape(-1, 1))
    ce_slope, ce_intercept = ce_model.coef_[0], ce_model.intercept_
    ce_r2 = r2_score(log_ce, ce_pred_log)

    # ---------------- -log(indicator) 拟合 ----------------
    log_rank_loss = np.log10(rank_loss_vals + 1e-12)
    rank_loss_model = LinearRegression()
    rank_loss_model.fit(logS.reshape(-1, 1), log_rank_loss)
    rank_loss_pred_log = rank_loss_model.predict(logS.reshape(-1, 1))
    rank_loss_slope, rank_loss_intercept = rank_loss_model.coef_[0], rank_loss_model.intercept_
    rank_loss_r2 = r2_score(log_rank_loss, rank_loss_pred_log)

    # ---------------- 双y轴 ----------------
    # ax1.set_ylim(0.2, 0.5)
    ax1.set_ylim(1.8, 5)
    ax2 = ax1.twinx()
    ax2.set_ylim(0.47, 1.13)

    # ---------------- 绘制原始数据 ----------------
    l1, = ax1.plot(S_vals, ce_loss_vals,
                   marker='o', markersize=4, linewidth=1.8, color=ce_color, alpha=0.9,
                   label="cross-entropy loss")
    l2, = ax2.plot(S_vals, rank_loss_vals,
                   marker='s', markersize=4, linewidth=1.8, color=rank_color, alpha=0.9,
                   label="$-\log \\text{RBP}_1 $")

    # ---------------- 绘制拟合曲线（灰色虚线，区分样式） ----------------
    l3, = ax1.plot(S_vals, 10**ce_pred_log,
                   linestyle='--', linewidth=2, color="gray", alpha=0.8,
                   label=f"Abs fit: slope = {ce_slope:.3f}, R²={ce_r2:.3f}")
    l4, = ax2.plot(S_vals, 10**rank_loss_pred_log,
                   linestyle=':', linewidth=2, color="gray", alpha=0.8,
                   label=f"Rel fit: slope = {rank_loss_slope:.3f}, R²={rank_loss_r2:.3f}")

    # ---------------- 坐标轴设置 ----------------
    ax1.set_xscale("log")
    ax1.set_xlabel("Model Size S", fontsize=14)

    ax1.set_yscale("log")
    ax1.set_ylabel("cross-entropy loss", fontsize=14, color="black")
    ax1.tick_params(axis='y', labelcolor="black")

    ax2.set_yscale("log")
    ax2.set_ylabel("$-\log \\text{RBP}_1$", fontsize=14, color="black")
    ax2.tick_params(axis='y', labelcolor="black")

    # ---------------- 合并图例 ----------------
    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, fontsize=9, loc='upper right', frameon=True,
               facecolor='white', edgecolor='gray', framealpha=0.9)

    ax1.set_title(f"Scaling Law (topk={topk})", fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig('assets/CE_and_Ranking_Loss_vs_S.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_F_vs_S(model_sizes, topk_list=[1, 5, 10, 50, 100, 500, 2500, 10000], seq_N_list=[], cmap_name="magma"):
    x = np.log(model_sizes).reshape(-1, 1)
    S = model_sizes

    # 创建大图：2x4 排版
    fig, axes = plt.subplots(2, 4, figsize=(30, 12))
    axes = axes.flatten()

    # 灵活调色盘
    try:
        cmap = plt.get_cmap(cmap_name)   # 支持 inferno, viridis, crest 等
    except:
        cmap = plt.cm.inferno  # fallback

    norm = LogNorm(vmin=min(seq_N_list), vmax=max(seq_N_list) * 2)

    for t_idx, topk in enumerate(topk_list):
        metrics = collect_metrics(topk)
        ax = axes[t_idx]

        alpha_list = []
        r2_list = []

        for n in seq_N_list:
            key = f"F_N={n}"
            if key not in metrics:
                continue

            F_vals = np.array(metrics[key])
            F_vals = np.clip(F_vals, 1e-12, 1 - 1e-12)

            # 线性化拟合
            y = np.log(-np.log(F_vals))
            model = LinearRegression()
            model.fit(x, y)
            y_pred_fit = model.predict(x)

            slope = model.coef_[0]
            intercept = model.intercept_
            alpha_CE = -slope
            alpha_list.append(alpha_CE)
            C = np.exp(intercept)

            r2 = r2_score(y, y_pred_fit)
            r2_list.append(r2)

            # 映射颜色（对数刻度）
            color = cmap(norm(n))

            # 实际点
            ax.plot(x, F_vals, 'o', markersize=5, color=color)

            # 拟合曲线
            y_pred_curve = np.exp(-C * S**(-alpha_CE))
            ax.plot(np.log(S), y_pred_curve, '-', color=color, alpha=0.6, linewidth=2)

        # 平均 alpha 与 R²
        avg_alpha = np.mean(alpha_list) if alpha_list else float('nan')
        avg_r2 = np.mean(r2_list) if r2_list else float('nan')

        # 图例显示 -log p_N ~ N * S^-alpha
        legend_text = f"$-\\log p_N \\propto N \\cdot S^{{-{avg_alpha:.3f}}}$\nR²={avg_r2:.3f}"
        ax.text(0.05, 0.95, legend_text,
                transform=ax.transAxes, fontsize=14,
                ha='left', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=1.0))

        # 子图美化
        ax.set_xlabel("model size S", fontsize=14)
        ax.set_ylabel(r"$p_N$", fontsize=14)
        ax.set_title(f"topk={topk}", fontsize=14, pad=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        # 横轴科学记数
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda val, _: f"$10^{{{int(np.round(np.log10(np.exp(val))))}}}$")
        )

    # 渐变色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, location="right")
    cbar.set_label("Sequence length N", fontsize=14)

    # 对数刻度刻度
    exp_min = int(np.floor(np.log10(min(seq_N_list))))
    exp_max = int(np.ceil(np.log10(max(seq_N_list) * 2)))
    ticks = [10**i for i in range(exp_min, exp_max+1)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"$10^{i}$" for i in range(exp_min, exp_max+1)])

    # fig.suptitle("Scaling Law: $p_N$ vs model size S (different topk)", fontsize=18)
    plt.savefig('assets/F_vs_S.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_p_gt_in_topk_R2_and_slope_vs_topk(model_sizes, topk_list):
    # 只考虑 Pythia 模型
    model_sizes = np.array([extract_size(name) for name in pythia_models])
    dataset_markers = ["o", "s", "^", "D"]  
    # 为不同 dataset 选择不同的渐变色系，并从中挑选一个代表颜色
    cmap_list = ["magma"]
    positions = [0.15, 0.3, 0.5, 0.7, 0.85]  # 预设取色点，差异较大
    palette = []
    for i, ds in enumerate(dataset_names):
        cmap_name = cmap_list[i % len(cmap_list)]
        cmap = sns.color_palette(cmap_name, as_cmap=True)
        pos = positions[i % len(positions)]
        palette.append(cmap(pos))

    # ========== 图1: R² vs topk ==========
    plt.figure(figsize=(7, 5))
    ax_r2 = plt.gca()

    for i, dataset_name in enumerate(dataset_names):
        R2_list = []
        for topk in topk_list:
            metrics = collect_metrics(topk, dataset_name=dataset_name, base_result_dir="eval_gt_in_topk",considered_model_names=pythia_models)
            r_vals = np.array(metrics["mean_indicator"])
            S_vals = np.array(model_sizes)

            logr_vals = np.log10(-np.log(r_vals) + 1e-12)
            logS_vals = np.log10(S_vals)

            model = LinearRegression()
            print(logr_vals)
            model.fit(logS_vals.reshape(-1, 1), logr_vals)
            
            y_pred = model.predict(logS_vals.reshape(-1, 1))

            r2 = r2_score(logr_vals, y_pred)
            R2_list.append(r2)

        ax_r2.plot(topk_list, R2_list,
                   color=palette[i], linewidth=1.7, alpha=0.9, zorder=2,
                   label=dataset_short[dataset_name])
        ax_r2.scatter(topk_list, R2_list,
                      color=palette[i], s=15, marker=dataset_markers[i], zorder=3)

    ax_r2.set_xscale("log")
    ax_r2.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"$10^{{{int(np.log10(val))}}}$"))
    ax_r2.set_xlabel("topk", fontsize=13)
    ax_r2.set_ylabel("$R^2$", fontsize=13)
    ax_r2.set_title("Fitting $-\log RBP_k$ - model size S: $R^2$ vs topk", fontsize=14, pad=10)
    ax_r2.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax_r2.legend(frameon=True, fontsize=10,
                 facecolor='white', edgecolor='gray', framealpha=0.9,
                 loc='best')

    plt.tight_layout()
    plt.savefig('assets/indicator-S_R2_vs_topk.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 图2: slope vs topk ==========
    
    plt.figure(figsize=(7, 5))
    ax_slope = plt.gca()

    for i, dataset_name in enumerate(dataset_names):
        slope_list = []
        for topk in topk_list:
            metrics = collect_metrics(topk, dataset_name=dataset_name, base_result_dir="eval_gt_in_topk",considered_model_names=pythia_models)
            r_vals = np.array(metrics["mean_indicator"])
            S_vals = np.array(model_sizes)

            logr_vals = np.log10(-np.log(r_vals) + 1e-12)
            logS_vals = np.log10(S_vals)

            model = LinearRegression()
            model.fit(logS_vals.reshape(-1, 1), logr_vals)

            slope = model.coef_[0]
            slope_list.append(slope)

        ax_slope.plot(topk_list, slope_list,
                      color=palette[i], linewidth=1.7, alpha=0.9, zorder=2,
                      label=dataset_short[dataset_name])
        ax_slope.scatter(topk_list, slope_list,
                         color=palette[i], s=15, marker=dataset_markers[i], zorder=3)
    
    # cross-entropy loss 数据
    ce_losses = {
        "wikimedia/wikipedia": -0.087,
        "openai/openai_humaneval": -0.102,
        "hotpotqa/hotpot_qa": -0.050,
        "isaacus/open-australian-legal-corpus": -0.077
    }

    # 绘制每个数据集的水平虚线
    for i, dataset_name in enumerate(dataset_names):
        ce_value = ce_losses[dataset_name]
        ax_slope.axhline(y=ce_value, color=palette[i], linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
        ax_slope.plot([], [], color=palette[i], linestyle="--", label=f"cross-entropy loss ({dataset_short[dataset_name]})")

    ax_slope.set_xscale("log")
    ax_slope.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"$10^{{{int(np.log10(val))}}}$"))
    ax_slope.set_xlabel("topk", fontsize=13)
    ax_slope.set_ylabel("Fitting slope", fontsize=13)
    ax_slope.set_title("Fitting $-\log RBP_k$ - model size S: slope vs topk", fontsize=14, pad=10)
    ax_slope.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax_slope.legend(frameon=True, fontsize=10,
                    facecolor='white', edgecolor='gray', framealpha=0.9,
                    loc='best')

    plt.tight_layout()
    plt.savefig('assets/indicator-S_slope_vs_topk.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_scale_multiple_datasets(model_sizes, metric_name, topk_list=[1]):
    # 每个系列只挑一个固定颜色
    series_colors = {
        "Pythia": sns.color_palette("crest", 10)[4],   
        "OPT": sns.color_palette("inferno", 10)[5],    
        "GPT2": sns.color_palette("viridis", 10)[6],   
        "Qwen": sns.color_palette("magma", 10)[3],     
    }

    # 每个系列一个 marker
    series_markers = {
        "Pythia": "o",
        "OPT": "s",
        "GPT2": "^",
        "Qwen": "D",
    }

    num_datasets = len(dataset_names)
    num_topk = len(topk_list)
    nrows = num_topk
    ncols = num_datasets

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols + 2, 3.5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    # 统一收集图例句柄
    legend_handles = []
    legend_labels = []

    for d_idx, dataset_name in enumerate(dataset_names):
        # 获取该 dataset 的系列信息
        series_array = collect_metrics_for_series(1, dataset_name=dataset_name, base_result_dir="eval_gt_in_topk")["series"]

        for t_idx, topk in enumerate(topk_list):
            metrics = collect_metrics_for_series(topk, dataset_name=dataset_name, base_result_dir="eval_gt_in_topk")
            ax = axes[t_idx, d_idx]

            r_vals = np.array(metrics[metric_name])
            S_vals = np.array(model_sizes)

            # 横纵轴 log 转换
            if metric_name == "mean_ce_loss":
                y_vals = r_vals
            else:
                y_vals = -np.log(r_vals)

            for series in np.unique(series_array):
                idxs = np.where(series_array == series)[0]
                S_series = S_vals[idxs]
                y_series = y_vals[idxs]

                # 绘制散点和线
                ax.scatter(
                    S_series,
                    y_series,
                    color=series_colors.get(series, "black"),
                    s=50,
                    alpha=0.9,
                    marker=series_markers.get(series, "o"),
                    linewidth=0.8
                )

                # 拟合曲线
                logS_series = np.log10(S_series)
                logy_series = np.log10(y_series + 1e-12)
                model = LinearRegression()
                model.fit(logS_series.reshape(-1,1), logy_series)
                print(series, dataset_name, model.coef_[0])
                y_pred_log = model.predict(logS_series.reshape(-1,1))

                line, = ax.plot(
                    S_series,
                    10**y_pred_log,
                    color=series_colors.get(series, "black"),
                    linewidth=2.0,
                    alpha=0.85,
                    # marker=series_markers.get(series, "o")
                )

                # 收集图例信息
                if series not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(series)

            # 坐标轴与标题
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Model Size S", fontsize=14)
            ax.set_ylabel("$-\log RBP_k$" if metric_name != "mean_ce_loss" else metric_name, fontsize=14, fontweight='bold')
            ax.set_title(f"{dataset_short[dataset_name]}, topk={topk}", fontsize=14, pad=8)
            ax.grid(True, linestyle='--', alpha=0.4)

    # 右侧统一图例
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center right',
        fontsize=14,
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        bbox_to_anchor=(0.93, 0.5)  
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 留出右侧空间给图例
    plt.savefig(f'assets/{metric_name}_vs_S_by_series_multiple_datasets_large.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_topk_ablation(metric_name, topk_list=[1,5,10,50,100,500]):
    # 只考虑 Pythia 模型
    model_sizes = np.array([extract_size(name) for name in pythia_models])
    num_topk = len(topk_list)
    nrows, ncols = 2, 3  # 2x3 子图布局
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    axes = np.array(axes).reshape(-1)

    # 准备颜色
    cmap_list = ["magma"]
    positions = [0.15, 0.3, 0.85, 0.5, 0.7]
    palette = []
    for i, ds in enumerate(dataset_names):
        cmap_name = cmap_list[i % len(cmap_list)]
        cmap = sns.color_palette(cmap_name, as_cmap=True)
        pos = positions[i % len(positions)]
        palette.append(cmap(pos))

    for t_idx, topk in enumerate(topk_list):
        ax = axes[t_idx]
        for d_idx, dataset_name in enumerate(dataset_names):
            metrics = collect_metrics_for_series(topk, dataset_name=dataset_name, base_result_dir="eval_gt_in_topk", considered_model_names=pythia_models)
            r_vals = np.array(metrics[metric_name])
            S_vals = np.array(model_sizes)

            if metric_name == "mean_ce_loss":
                y_vals = r_vals
            else:
                y_vals = -np.log(r_vals)

            # 绘制散点
            ax.scatter(
                S_vals,
                y_vals,
                color=palette[d_idx],
                s=30,
                alpha=0.9,
                marker='o',
                linewidth=0.8,
                # label=f"{dataset_name} data"
            )

            # 拟合曲线
            logS = np.log10(S_vals)
            logy = np.log10(y_vals + 1e-12)
            model = LinearRegression()
            model.fit(logS.reshape(-1,1), logy)
            y_pred_log = model.predict(logS.reshape(-1,1))
            slope = model.coef_[0]
            r2 = r2_score(logy, y_pred_log)

            ax.plot(
                S_vals,
                10**y_pred_log,
                color=palette[d_idx],
                linewidth=2.0,
                alpha=0.85,
                label=f"{dataset_short[dataset_name]}: slope={slope:.3f}, R²={r2:.3f}"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Model Size S", fontsize=14)
        ax.set_ylabel("$-\log p_R$" if metric_name != "mean_ce_loss" else metric_name,
                      fontsize=14, fontweight='bold')
        ax.set_title(f"topk={topk}", fontsize=14, pad=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9, loc='lower left')

    # 隐藏多余子图
    for i in range(num_topk, nrows * ncols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(f'assets/{metric_name}_topk_ablation_pythia.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    topk_list = [1, 2, 3, 5, 7, 10, 18, 30, 50, 78, 100, 200, 336, 500, 886, 1438, 2500, 3793, 6158]
    # topk_list += [10000, 20000, 30000, 50000]
    # topk_list += [10000, 20000, 30000]
    topk_list += [10000]
    # seq_N_list = list(range(1, 11)) + [20, 30, 40, 50, 100, 200, 500]
    # seq_N_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    
    # plot_logr_vs_logS(model_sizes)
    # plot_p_gt_in_topk_R2_and_slope_vs_topk(model_sizes, topk_list)
    # plot_r_vs_position()
    # plot_F_vs_S(model_sizes=model_sizes, seq_N_list=seq_N_list)
    # plot_logN_vs_logS(model_sizes)
    # plot_logN_R2_vs_topk(model_sizes, topk_list)
    # plot_simulated_N_vs_S(model_sizes)
    # plot_F_R2_vs_topk(model_sizes, topk_list, seq_N_list=seq_N_list)
    # plot_N_vs_invF(model_sizes=model_sizes, topk=10000, seq_N_list=seq_N_list)
    # plot_indicator_vs_S(model_sizes, topk_list=topk_list)
    plot_CE_and_rankloss_vs_S(model_sizes)
    # plot_scale(model_sizes, "mean_indicator_multiply_p_gt_divide_sum_of_topk_p", topk_list)
    # plot_scale_multiple_datasets(model_sizes, "mean_ce_loss", topk_list=[1]) 
    # plot_topk_ablation("mean_indicator")
    # plot_logCE_vs_logS("wikimedia/wikipedia", model_sizes,topk=1)