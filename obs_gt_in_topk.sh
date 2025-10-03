export HF_HOME="/data-share/guest/yuebaoqing/.hf_cache"
export CUDA_VISIBLE_DEVICES=0
dataset="wikimedia/wikipedia"
data_num=1000
output_prefix="eval_gt_in_topk"

# ========= model series =========
models=(
    "openai-community/gpt2"
    "openai-community/gpt2-large"
    "openai-community/gpt2-medium"
    "openai-community/gpt2-xl"
)

for model in "${models[@]}"; do
    label_model=$model # use their own tokenizer
    data_dir="gen_dataset/corpus/${dataset}/${label_model}/test"
    output_dir="$output_prefix/${dataset}/${model}"

    python src/obs_gt_in_topk.py \
        --model "$model" \
        --data_dir "$data_dir" \
        --output_dir "$output_dir" \
        --data_num "$data_num"
done

