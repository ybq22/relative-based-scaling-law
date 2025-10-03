import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

gpt2_models = [
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
]
eval_model_names = gpt2_models # you can add other models here

def extract_size(model_name, count_json_path="assets/param_count.json"):
    with open(count_json_path, "r") as f:
        model_param_count = json.load(f)
    size = model_param_count[model_name]["non_embedding_params"]
    return size

def load_metrics(path): # load metrics from certain path
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                metrics = json.load(f)
            metrics = {k: v for k, v in metrics.items()}
            return metrics
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
            return {}
    else:
        print(f"⚠️ Missing: {path}")
        return {}

def collect_metrics(topk, dataset_name="wikimedia/wikipedia", base_result_dir="eval_gt_in_topk",considered_model_names=eval_model_names): # re-organize the collected metrics
    collected = {}
    for eval_model in considered_model_names:
        path = os.path.join(
            base_result_dir,
            dataset_name,
            eval_model,
            f"top{topk}",
            "metrics.json"
        )
        metrics = load_metrics(path)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                collected.setdefault(k, []).append(v)
            else:
                collected.setdefault(k, []).append(np.nan)

    # transform to numpy array
    collected = {k: np.array(v, dtype=float) for k, v in collected.items()}
    return collected


def plot_ce_vs_S(model_sizes, metrics, topk=1, save_path="assets/CE_vs_S.png"):
    S_vals = np.array(model_sizes)
    ce_loss_vals = np.array(metrics["mean_ce_loss"])
    logS = np.log10(S_vals)

    log_ce = np.log10(ce_loss_vals + 1e-12)
    ce_model = LinearRegression()
    ce_model.fit(logS.reshape(-1, 1), log_ce)
    ce_pred_log = ce_model.predict(logS.reshape(-1, 1))
    ce_slope, ce_r2 = ce_model.coef_[0], r2_score(log_ce, ce_pred_log)

    plt.figure()
    plt.plot(S_vals, ce_loss_vals, marker='o', label="cross-entropy loss")
    plt.plot(S_vals, 10**ce_pred_log, linestyle='--', label=f"fit slope={ce_slope:.3f}, R²={ce_r2:.3f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Model Size S")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Scaling of CE Loss (topk={topk})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_rbp_vs_S(model_sizes, metrics, topk=1, save_path="assets/RBP_vs_S.png"):
    S_vals = np.array(model_sizes)
    indicator_vals = np.array(metrics[f"RBP_{topk}"])
    rank_loss_vals = -np.log(indicator_vals + 1e-12)
    logS = np.log10(S_vals)

    # 拟合 log-log 关系
    log_rank_loss = np.log10(rank_loss_vals + 1e-12)
    model = LinearRegression()
    model.fit(logS.reshape(-1, 1), log_rank_loss)
    pred_log = model.predict(logS.reshape(-1, 1))
    slope, r2 = model.coef_[0], r2_score(log_rank_loss, pred_log)

    plt.figure()
    plt.plot(S_vals, rank_loss_vals, marker='s', label="-log RBP")
    plt.plot(S_vals, 10**pred_log, linestyle='--', label=f"fit slope={slope:.3f}, R²={r2:.3f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Model Size S")
    plt.ylabel("-log RBP")
    plt.title(f"Scaling of -log RBP (topk={topk})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    metrics = collect_metrics(topk=1) # or other configurations
    model_sizes = np.array([extract_size(name) for name in eval_model_names])
    plot_ce_vs_S(model_sizes, metrics)
    plot_rbp_vs_S(model_sizes, metrics)