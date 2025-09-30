import numpy as np
import json,os
# 所有模型列表
# 四个模型系列
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

# 拼接成最终数组
eval_model_names = pythia_models + opt_models + gpt2_models + qwen_models
eval_model_names = pythia_models

def get_model_series(model_name):
    if model_name in pythia_models:
        return "Pythia"
    elif model_name in opt_models:
        return "OPT"
    elif model_name in gpt2_models:
        return "GPT2"
    elif model_name in qwen_models:
        return "Qwen"
    else:
        return "Unknown"

# 定义提取模型大小的函数
def extract_size(model_name, count_json_path="/data-share/guest/yuebaoqing/dr/language_modeling/param_count/count.json"):
    with open(count_json_path, "r") as f:
        model_param_count = json.load(f)
    size = model_param_count[model_name]["non_embedding_params"]
    return size

# 构建 model_sizes 列表
model_sizes = np.array([extract_size(name) for name in eval_model_names])


dataset_names = [
    "wikimedia/wikipedia",
    # "openai/gsm8k",
    "openai/openai_humaneval",
    "hotpotqa/hotpot_qa",
    "isaacus/open-australian-legal-corpus"
]


dataset_short = {
    "wikimedia/wikipedia": "Wiki",
    "openai/openai_humaneval": "HumanEval",
    "hotpotqa/hotpot_qa": "HotpotQA",
    "isaacus/open-australian-legal-corpus": "AusLegal"
}



def load_metrics(path):
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

def collect_metrics(topk, dataset_name="wikimedia/wikipedia", base_result_dir="eval_N",considered_model_names=eval_model_names):
    """收集所有模型的 metrics，并整理成 {metric_name: np.array([...])}"""
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

    # 转成 numpy array
    collected = {k: np.array(v, dtype=float) for k, v in collected.items()}
    return collected

def collect_metrics_for_series(topk, dataset_name="wikimedia/wikipedia", base_result_dir="eval_N", considered_model_names=eval_model_names):
    """收集所有模型的 metrics，并整理成 {metric_name: np.array([...]), 'series': np.array([...])}"""
    collected = {}
    series_list = []

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
        # 收集模型系列信息
        series_list.append(get_model_series(eval_model))

    # 转成 numpy array
    collected = {k: np.array(v, dtype=float) for k, v in collected.items()}
    collected["series"] = np.array(series_list)
    return collected

