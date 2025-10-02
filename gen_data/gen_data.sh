#!/usr/bin/env bash

# 定义中断处理函数
cleanup() {
    echo "中断信号收到，终止所有子进程..."
    pkill -P $$
    exit 1
}
trap cleanup SIGINT SIGTERM

# 参数配置
num_sequences=1000
seq_len=512

models=(
    "EleutherAI/pythia-14m"
    "facebook/opt-125m"
    "openai-community/gpt2-medium"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"
)

batch_sizes=(
    16
    16
    8
    4
    4
    2
    1
    1
)

datasets=(
    # "wikimedia/wikipedia"
    # "openai/gsm8k"
    # "openai/openai_humaneval"
    # "hotpotqa/hotpot_qa"
    # "isaacus/open-australian-legal-corpus"
    # "allenai/c4"
    "monology/pile-uncopyrighted/Github"
)

# GPU列表（假设你有8块GPU）
gpus=(1)
num_gpus=${#gpus[@]}
max_parallel=$num_gpus

# 生成任务
for temperature in 1; do
    echo "=== Temperature = $temperature ==="

    for i in "${!models[@]}"; do
        model="${models[$i]}"
        batch_size="${batch_sizes[$i]}"
        gpu_index=$((i % num_gpus))
        gpu="${gpus[$gpu_index]}"

        for dataset in "${datasets[@]}"; do
            output_dir="./corpus/${dataset}/${model}"
            echo "Launching model $model on GPU $gpu with batch size $batch_size on dataset $dataset"

            (
            # export CUDA_VISIBLE_DEVICES=$gpu
            export CUDA_VISIBLE_DEVICES=0,1,5,6
            export HF_ENDPOINT=https://hf-mirror.com
            export CUBLAS_WORKSPACE_CONFIG=:4096:8
            export HF_HOME="/data-share/guest/yuebaoqing/.hf_cache"
            # export export HF_HUB_OFFLINE="1"


            python ./gen_data.py \
                --model "$model" \
                --seq_len "$seq_len" \
                --num_sequences "$num_sequences" \
                --batch_size "$batch_size" \
                --output_dir "$output_dir" \
                --temperature $temperature \
                --dataset "$dataset" \
                --overwrite \
                --deterministic

            echo "Finished model $model on GPU $gpu with dataset $dataset"
            )
        done
    done
done

echo "✅ 所有 generation 任务已完成。"
