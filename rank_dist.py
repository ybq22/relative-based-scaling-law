import os
import json
import matplotlib.pyplot as plt
import powerlaw
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_ranks(base_dir, model):
    """load ranks.json"""
    all_ranks = {}
    rank_file = os.path.join(base_dir, model, "top1", "ranks.json")
    if not os.path.exists(rank_file):
        print(f"⚠️ skipping {model}, file not exists: {rank_file}")
    with open(rank_file, "r") as f:
        data = json.load(f)
    ranks = [item[0] for item in data]
    return ranks


def fit_distributions(ranks, xmin=1):
    fit = powerlaw.Fit(ranks, discrete=True, verbose=False, xmin=xmin)
    R, p = fit.distribution_compare('power_law', 'lognormal')
    return fit, R, p


def plot_single_model(ranks, model_name, output_path, xmin=1):
    fit, R, p = fit_distributions(ranks, xmin=xmin)
    alpha = fit.power_law.alpha
    mu = fit.lognormal.mu
    sigma = fit.lognormal.sigma

    plt.figure()
    fit.plot_pdf(label="data")
    fit.power_law.plot_pdf(label=f"power law α={alpha:.3f}")
    fit.lognormal.plot_pdf(label=f"lognormal μ={mu:.3f}, σ={sigma:.3f}")

    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title(f"{model_name} (R={R:.3f}, p={p:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return {
        "alpha": alpha,
        "R_power_lognormal": R,
        "p_value": p,
        "lognormal_mu": mu,
        "lognormal_sigma": sigma
    }


if __name__ == "__main__":
    model_name = "openai-community/gpt2"
    dataset_name = "wikimedia/wikipedia"
    ranks = load_ranks(base_dir=f"eval_gt_in_topk/{dataset_name}", model=model_name)
    res = plot_single_model(ranks, model_name, f"assets/{model_name.split('/')[-1]}_rank_dist.png")
    print(res)
    
    
