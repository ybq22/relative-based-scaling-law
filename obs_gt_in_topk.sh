#!/usr/bin/env bash

# ========= 基础设置 =========
GPUS=(0 1 2 3 5)  # 可根据需要修改
MULTI_GPU_DEVICES="0,1,2,3,5"
N_GPUS=${#GPUS[@]}

data_num=1000
export HF_HOME="/data-share/guest/yuebaoqing/.hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"

TASK_FILE="/tmp/obs_gt_in_topk_queue.txt"
LOCK_FILE="/tmp/obs_gt_in_topk_task.lock"

# ========== 中断处理 ==========
cleanup() {
    echo "⚠️ 中断信号收到，终止所有子进程..."
    pkill -P $$
    rm -f "$TASK_FILE" "$LOCK_FILE"
    exit 1
}
trap cleanup SIGINT SIGTERM

# ========== 模型列表 ==========
eval_model_names=(
    "EleutherAI/pythia-14m"
    "EleutherAI/pythia-31m"
    "EleutherAI/pythia-70m"
    "EleutherAI/pythia-160m"
    "EleutherAI/pythia-410m"
    "EleutherAI/pythia-1b"
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-2.8b"
    "EleutherAI/pythia-6.9b"

    "facebook/opt-125m"
    "facebook/opt-350m"
    "facebook/opt-1.3b"
    "facebook/opt-2.7b"
    "facebook/opt-6.7b"

    "openai-community/gpt2"
    "openai-community/gpt2-large"
    "openai-community/gpt2-medium"
    "openai-community/gpt2-xl"

    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-7B"
)
output_prefix="eval_gt_in_topk"
# ========== 数据集列表 ==========
datasets=(
    # "wikimedia/wikipedia"
    # "openai/gsm8k"
    # "openai/openai_humaneval"
    # "hotpotqa/hotpot_qa"
    # "isaacus/open-australian-legal-corpus"
    "monology/pile-uncopyrighted/Github"
    "allenai/c4"
)

# ========== 多卡大模型列表 ==========
multi_gpu_models=("Qwen/Qwen2.5-14B" "EleutherAI/pythia-12b")

get_label_model() {
    model_name=$1
    if [[ $model_name == EleutherAI/* ]]; then
        echo "EleutherAI/pythia-14m"
    elif [[ $model_name == facebook/* ]]; then
        echo "facebook/opt-125m"
    elif [[ $model_name == openai-community/* ]]; then
        echo "openai-community/gpt2-medium"
    elif [[ $model_name == Qwen/* ]]; then
        # Qwen 系列：label model 就是自己
        echo "$model_name"
    else
        echo "未知系列: $model_name" >&2
        exit 1
    fi
}

# ========== 写入任务队列（只写单卡小模型） ==========
> "$TASK_FILE"
for dataset in "${datasets[@]}"; do
    for eval_model in "${eval_model_names[@]}"; do
        # 跳过多卡模型
        if [[ " ${multi_gpu_models[@]} " =~ " ${eval_model} " ]]; then
            continue
        fi
        echo "${dataset}|${eval_model}" >> "$TASK_FILE"
    done
done

# ========== 获取 batch size ==========
get_bs() {
    case "$1" in
        # Pythia series models
        *pythia-14m) echo 64 ;;
        *pythia-31m) echo 32 ;;
        *pythia-70m) echo 32 ;;
        *pythia-160m) echo 32 ;;
        *pythia-410m) echo 32 ;;
        *pythia-1b) echo 8 ;;
        *pythia-1.4b) echo 4 ;;
        *pythia-2.8b) echo 4 ;;
        *pythia-6.9b) echo 2 ;;
        *pythia-12b) echo 1 ;;
        
        # Facebook/OPT series models
        *opt-125m) echo 64 ;;
        *opt-350m) echo 32 ;;
        *opt-1.3b) echo 8 ;;
        *opt-2.7b) echo 4 ;;
        *opt-6.7b) echo 2 ;;
        *opt-13b) echo 1 ;;

        # OpenAI/GPT-2 series models
        *gpt2) echo 32 ;;
        *gpt2-medium) echo 16 ;;
        *gpt2-large) echo 8 ;;
        *gpt2-xl) echo 4 ;;
        
        # Qwen2.5 series models
        *Qwen2.5-0.5B) echo 16 ;;
        *Qwen2.5-1.5B) echo 8 ;; # Use 16 for Qwen2.5-1.5B based on reported usage
        *Qwen2.5-3B) echo 4 ;;
        *Qwen2.5-7B) echo 1 ;;  # For standard hardware, reduce batch size. Example uses micro-batch 2 for 4 GPUs
        *qwen2.5-14B) echo 1 ;;
        *Qwen2.5-32B) echo 1 ;;
        *Qwen2.5-72B) echo 1 ;; # Likely requires multiple GPUs or deep quantization
        
        # Default case for any other model
        *) echo "❌ Unknown model: $1" >&2; exit 1 ;;
    esac
}


# ========== 获取任务 ==========
get_next_task() {
    local line=""
    {
        flock -x 200
        line=$(head -n 1 "$TASK_FILE")
        if [[ -n "$line" ]]; then
            tail -n +2 "$TASK_FILE" > "${TASK_FILE}.tmp" && mv "${TASK_FILE}.tmp" "$TASK_FILE"
        fi
        echo "$line"
    } 200>"$LOCK_FILE"
}

# ========== 多卡大模型处理 ==========
for model in "${multi_gpu_models[@]}"; do
    for dataset in "${datasets[@]}"; do
        # label_model=$model # for Qwen only
        label_model=$(get_label_model "$model")
        data_dir="gen_dataset/corpus/${dataset}/${label_model}/test"
        output_dir="$output_prefix/${dataset}/${model}"

        echo "🚀 [Multi-GPU] 开始任务: $model on dataset $dataset"
        export CUDA_VISIBLE_DEVICES=$MULTI_GPU_DEVICES

        python src/obs_gt_in_topk.py \
            --model "$model" \
            --data_dir "$data_dir" \
            --output_dir "$output_dir" \
            --data_num "$data_num"

        echo "✅ [Multi-GPU] 完成任务: $model on dataset $dataset"
    done
done

# ========== 单 GPU worker ==========
gpu_worker() {
    local cuda_id=$1
    while true; do
        task=$(get_next_task)
        if [[ -z "$task" ]]; then
            echo "✅ [GPU $cuda_id] 无任务，退出"
            break
        fi

        IFS='|' read -r dataset eval_model <<< "$task"
        bs=$(get_bs "$eval_model")
        grad=$((1024 / bs))
        label_model=$(get_label_model "$eval_model")

        data_dir="gen_dataset/corpus/${dataset}/${label_model}/test"
        output_dir="$output_prefix/${dataset}/${eval_model}"

        echo "🚀 [GPU $cuda_id] 开始任务: $eval_model on dataset $dataset (bs=$bs)"
        export CUDA_VISIBLE_DEVICES="$cuda_id"

        python src/obs_gt_in_topk.py \
            --model "$eval_model" \
            --data_dir "$data_dir" \
            --output_dir "$output_dir" \
            --data_num "$data_num"

        echo "✅ [GPU $cuda_id] 完成任务: $eval_model on dataset $dataset"
    done
}

# ========== 启动单卡 GPU worker ==========
for cuda_id in "${GPUS[@]}"; do
    gpu_worker "$cuda_id" &
done

wait
echo "✅ 所有单卡模型任务完成！"



# ========== 清理 ==========
rm -f "$TASK_FILE" "$LOCK_FILE"
echo "🎉 所有 Raw Wiki obs GT 任务完成！"
