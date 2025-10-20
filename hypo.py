import os
import json
import matplotlib.pyplot as plt
import powerlaw
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import norm
import seaborn as sns

def extract_size(model_name, count_json_path="assets/param_count.json"):
    """Extract model size from parameter count file"""
    with open(count_json_path, "r") as f:
        model_param_count = json.load(f)
    size = model_param_count[model_name]["non_embedding_params"]
    return size

def load_ranks(base_dir, model):
    """load ranks.json"""
    rank_file = os.path.join(base_dir, model, "top1", "ranks.json")
    if not os.path.exists(rank_file):
        print(f"⚠️ skipping {model}, file not exists: {rank_file}")
        return None
    with open(rank_file, "r") as f:
        data = json.load(f)
    ranks = [item[0] for item in data]
    return ranks

def fit_distributions(ranks, xmin=1):
    """Fit power law and lognormal distributions using powerlaw library"""
    fit = powerlaw.Fit(ranks, discrete=True, verbose=False, xmin=xmin)
    R, p = fit.distribution_compare('power_law', 'lognormal')
    return fit, R, p

def plot_single_model(ranks, model_name, output_path, xmin=1):
    """Plot rank distribution for a single model"""
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
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return {
        "alpha": alpha,
        "R_power_lognormal": R,
        "p_value": p,
        "lognormal_mu": mu,
        "lognormal_sigma": sigma
    }

# Hypothesis function definitions
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

def analyze_multiple_models(base_dir, dataset_name, model_list):
    """
    Analyze rank distributions of multiple models and predict scaling laws
    
    Args:
        base_dir: Base directory
        dataset_name: Dataset name
        model_list: List of models
    """
    
    mus = []
    sigmas = []
    alphas = []
    model_sizes = []
    
    print("Analyzing model rank distributions...")
    
    # Step 1: Analyze rank distribution for each model
    for i, model_name in enumerate(model_list):
        print(f"Analyzing model {i+1}/{len(model_list)}: {model_name}")
        
        # Get model size
        try:
            model_size = extract_size(model_name)
            model_sizes.append(model_size)
        except Exception as e:
            print(f"Failed to get size for model {model_name}: {e}")
            continue
        
        # Load rank data
        ranks = load_ranks(base_dir=f"{base_dir}/{dataset_name}", model=model_name)
        if ranks is None:
            print(f"Skipping model {model_name}, failed to load rank data")
            continue
            
        # Save distribution plot
        output_path = f"assets/{model_name.split('/')[-1]}_rank_dist.png"
        
        result = plot_single_model(ranks, model_name, output_path)
        if result is None:
            continue
        
        mus.append(result["lognormal_mu"])
        sigmas.append(result["lognormal_sigma"])
        alphas.append(result["alpha"])
        
        print(f"  Model size: {model_size:,}")
        print(f"  Fitted parameters: μ={result['lognormal_mu']:.3f}, σ={result['lognormal_sigma']:.3f}, α={result['alpha']:.3f}")
    
    print(f"\nSuccessfully analyzed {len(mus)} models")
    if len(mus) > 0:
        print(f"μ value range: [{min(mus):.3f}, {max(mus):.3f}]")
        print(f"σ value range: [{min(sigmas):.3f}, {max(sigmas):.3f}]")
        print(f"Model size range: [{min(model_sizes):,}, {max(model_sizes):,}]")
    
    # Step 2: Use fitted parameters to predict scaling laws
    print("\nPredicting scaling laws...")
    
    p_ces, p_mlps = [], []
    for mu, sigma in zip(mus, sigmas):
        p_mlp, p_ce = pred_CE(sigma, mu)
        p_ces.append(p_ce)
        p_mlps.append(p_mlp)

    S_vals = np.array(model_sizes)
    ce_loss_vals = np.array(p_ces)   # Cross-entropy prediction
    mlp_vals    = np.array(p_mlps)   # -log(p1) prediction

    # Linear fitting in log coordinates
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

    print(f"Cross-entropy loss scaling slope: {ce_slope:.3f}, R² = {ce_r2:.3f}")
    print(f"-log(RBP₁) scaling slope: {mlp_slope:.3f}, R² = {mlp_r2:.3f}")

    # Plotting
    print("\nGenerating scaling law plots...")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax2.set_ylim(0.8, 1.9)

    # Color scheme
    colors = sns.color_palette("viridis", 5)
    ce_color = colors[3]   # Blue
    mlp_color = colors[0]  # Green-yellow

    # Original points and lines
    l1, = ax1.plot(S_vals, ce_loss_vals, marker='o', markersize=4, linewidth=1.8,
                   color=ce_color, alpha=0.9, label="Predicted cross-entropy loss")
    l2, = ax2.plot(S_vals, mlp_vals, marker='s', markersize=4, linewidth=1.8,
                   color=mlp_color, alpha=0.9, label="Predicted $-\\log(\\text{RBP}_1)$")

    # Fitted lines (gray dashed/dotted lines)
    l3, = ax1.plot(S_vals, 10**ce_pred_log, linestyle='--', linewidth=2,
                   color="gray", alpha=0.8, label=f"ce fit: slope={ce_slope:.3f}, R²={ce_r2:.3f}")
    l4, = ax2.plot(S_vals, 10**mlp_pred_log, linestyle=':', linewidth=2,
                   color="gray", alpha=0.8, label="$-\\log(\\text{RBP}_1)$" + f" fit: slope={mlp_slope:.3f}, R²={mlp_r2:.3f}")

    # Axis settings
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.set_xlabel("Model Size S", fontsize=14)
    ax1.set_ylabel("cross-entropy loss", fontsize=14, color="black")
    ax2.set_ylabel("$-\\log(\\text{RBP}_1)$", fontsize=14, color="black")

    ax1.tick_params(axis='y', labelcolor="black")
    ax2.tick_params(axis='y', labelcolor="black")

    # Combine legends
    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, fontsize=9, loc='upper right', frameon=True,
               facecolor='white', edgecolor='gray', framealpha=0.9)

    ax1.set_title(f"Simulated Scaling (topk=1) - {dataset_name}", fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig("CE_vs_logp1_scaling.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    
    return {
        "mus": mus,
        "sigmas": sigmas,
        "alphas": alphas,
        "model_sizes": model_sizes,
        "ce_slope": ce_slope,
        "ce_r2": ce_r2,
        "mlp_slope": mlp_slope,
        "mlp_r2": mlp_r2
    }


if __name__ == "__main__":
    # Example: Analyze single model
    print("=== Single Model Analysis Example ===")
    model_name = "openai-community/gpt2"
    dataset_name = "wikimedia/wikipedia"
    
    ranks = load_ranks(base_dir=f"eval_gt_in_topk/{dataset_name}", model=model_name)
    if ranks is not None:
        res = plot_single_model(ranks, model_name, f"assets/{model_name.split('/')[-1]}_rank_dist.png")
        if res is not None:
            print("Single model analysis results:")
            print(res)
    
    print("\n=== Multi-Model Analysis Example ===")
    # Example: Analyze multiple models (using GPT-2 series)
    model_list = [
        "openai-community/gpt2",
        "openai-community/gpt2-medium",
        "openai-community/gpt2-large",
        "openai-community/gpt2-xl"
    ]
    
    # Analyze multiple models and predict scaling laws
    results = analyze_multiple_models(
        base_dir="eval_gt_in_topk",
        dataset_name="wikimedia/wikipedia", 
        model_list=model_list
    )
    
    print("\n=== Final Results Summary ===")
    if results["mus"]:
        print(f"Cross-entropy loss scaling slope: {results['ce_slope']:.3f}")
        print(f"Cross-entropy loss fit quality: {results['ce_r2']:.3f}")
        print(f"-log(RBP₁) scaling slope: {results['mlp_slope']:.3f}")
        print(f"-log(RBP₁) fit quality: {results['mlp_r2']:.3f}")
    else:
        print("No models were successfully analyzed")
