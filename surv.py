import os
import json
import matplotlib.pyplot as plt
import powerlaw
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
import seaborn as sns

pythia_models = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]

opt_models = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
]

gpt2_models = [
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
]

qwen_models = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
]

dataset_names = [
    "wikimedia/wikipedia",
    # "openai/gsm8k",
    # "openai/openai_humaneval",
    # "hotpotqa/hotpot_qa",
    "isaacus/open-australian-legal-corpus"
]


dataset_short = {
    "wikimedia/wikipedia": "Wiki",
    "openai/openai_humaneval": "HumanEval",
    "hotpotqa/hotpot_qa": "HotpotQA",
    "isaacus/open-australian-legal-corpus": "AusLegal"
}

model_series = {
        "pythia": pythia_models,
        # "opt": opt_models,
        # "gpt2": gpt2_models,
        # "qwen": qwen_models,
    }

def load_ranks(base_dir, models):
    """加载所有模型的 ranks.json"""
    all_ranks = {}
    for model in models:
        rank_file = os.path.join(base_dir, model, "top1", "ranks.json")
        if not os.path.exists(rank_file):
            print(f"⚠️ 跳过 {model}, 文件不存在: {rank_file}")
            continue
        with open(rank_file, "r") as f:
            data = json.load(f)
        ranks = [item[0] for item in data]
        all_ranks[model] = ranks
    return all_ranks

def fit_distributions(ranks, xmin=2):
    """
    对给定 rank 数据进行 powerlaw 和 lognormal 拟合
    返回拟合对象和比较信息
    """
    fit = powerlaw.Fit(ranks, discrete=True, verbose=False, xmin=xmin)
    # 比较 powerlaw 和 lognormal
    R, p = fit.distribution_compare('power_law', 'lognormal')
    return fit, R, p

def plot_all_models(all_ranks, output_path, xmin=2):
    """绘制所有模型 rank 分布及拟合"""
    models = list(all_ranks.keys())
    n_rows, n_cols = 2, 5
    plt.figure(figsize=(5*n_cols, 4*n_rows))

    results = {}

    for i, model in enumerate(models, 1):
        ranks = all_ranks[model]
        fit, R, p = fit_distributions(ranks, xmin=xmin)
        alpha = fit.power_law.alpha
        mu = fit.lognormal.mu
        sigma = fit.lognormal.sigma

        ax = plt.subplot(n_rows, n_cols, i)
        fit.plot_pdf(color="b", marker="o", markersize=3, ax=ax, label="data", alpha=0.4)
        fit.power_law.plot_pdf(
            color="r", linestyle="--", ax=ax,
            label=f"power law α={alpha:.2f}"
        )
        fit.lognormal.plot_pdf(
            color="darkgreen", linestyle="-", linewidth=2, ax=ax,
            label=f"lognormal μ={mu:.2f}, σ={sigma:.2f}"
        )

        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model}\nR={R:.3f}, p={p:.3f}")
        ax.legend()
        ax.grid(True, which="major", linestyle="--", linewidth=0.5)

        results[model] = {
            "alpha": alpha,
            "xmin": fit.power_law.xmin,
            "R_power_lognormal": R,
            "p_value": p,
            "lognormal_mu": mu,
            "lognormal_sigma": sigma
        }

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return results

def extract_size(model_name, count_json_path="/data-share/guest/yuebaoqing/dr/language_modeling/param_count/count.json"):
    with open(count_json_path, "r") as f:
        model_param_count = json.load(f)
    size = model_param_count[model_name]["non_embedding_params"]
    return size

def collect_mu_over_sigma(base_dir, datasets, model_groups, xmin=2):
    """
    收集 lognormal μ/σ
    返回: dict[dataset][model] = {"mu_over_sigma": , "size": }
    """
    results = {ds: {} for ds in datasets}
    for ds in datasets:
        for group in model_groups:
            for model in group:
                rank_file = os.path.join(base_dir, ds, model, "top1", "ranks.json")
                if not os.path.exists(rank_file):
                    print('fiu')
                    continue
                with open(rank_file, "r") as f:
                    data = json.load(f)
                ranks = [item[0] for item in data]
                # print(ranks)
                fit = powerlaw.Fit(ranks, discrete=True, verbose=False, xmin=xmin)
                mu = fit.lognormal.mu
                sigma = fit.lognormal.sigma
                size = extract_size(model)
                results[ds][model] = {"mu_over_sigma": math.fabs(mu / (sigma+1e-12)), "size": size}
                print(ds, model, mu, sigma, size)
    return results

def plot_mu_over_sigma_pythia(results, dataset_names, dataset_short, model_series, output_path):
    plt.figure(figsize=(8, 6))
    series_name = "pythia"
    models = model_series[series_name]
    colors = sns.color_palette("viridis", 10)
    
    for idx, ds in enumerate(dataset_names):
        color = colors[0 + idx * 3]
        xs_raw, ys_raw = [], []
        for model in models:
            if model not in results[ds]:
                continue
            size = results[ds][model]["size"]
            val = results[ds][model]["mu_over_sigma"]
            xs_raw.append(size)
            ys_raw.append(val)
        if not xs_raw:
            continue

        # 拟合还是在 log 空间做
        xs_log = np.log10(xs_raw).reshape(-1, 1)
        ys_log = np.log10(ys_raw)
        reg = LinearRegression().fit(xs_log, ys_log)
        y_pred_log = reg.predict(xs_log)
        r2 = r2_score(ys_log, y_pred_log)

        # 拟合结果还原到原始空间
        y_pred_raw = 10 ** y_pred_log

        # 画散点（原始空间）
        plt.scatter(xs_raw, ys_raw, color=color, alpha=0.7)

        # 画拟合曲线（按 log 排序避免折线乱跳）
        order = np.argsort(xs_raw)
        plt.plot(np.array(xs_raw)[order], np.array(y_pred_raw)[order],
                 color=color, linestyle="--",
                 label=f"{dataset_short[ds]} (R²={r2:.2f})")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Model Size")
    plt.ylabel("Lognormal μ/σ")
    plt.title("Pythia: μ/σ Scaling Across Datasets")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    output_dir = "./assets/plot_sec5"
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有 dataset 的结果
    results = collect_mu_over_sigma(
        base_dir="./eval_gt_in_topk",   # 根目录
        datasets=dataset_names,         # 四个数据集
        model_groups=model_series.values(),
        xmin=2
    )

    # 只画 Pythia
    plot_mu_over_sigma_pythia(
        results,
        dataset_names,
        dataset_short,
        model_series,
        os.path.join(output_dir, "pythia_mu_over_sigma_scaling.pdf")
    )
    
    # plot_all_models(results,output_path="./a.pdf")
    
    
