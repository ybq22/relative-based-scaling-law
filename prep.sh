num_sequences=1000
seq_len=512
batch_size=4
dataset="wikimedia/wikipedia"

# ========= model series =========
models=(
    "openai-community/gpt2"
    "openai-community/gpt2-large"
    "openai-community/gpt2-medium"
    "openai-community/gpt2-xl"
)


export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8


for model in "${models[@]}"; do
    output_dir="./gen_dataset/corpus/$dataset/$model"
    python ./src/prep.py \
        --model "$model" \
        --seq_len "$seq_len" \
        --num_sequences "$num_sequences" \
        --batch_size "$batch_size" \
        --output_dir "$output_dir" \
        --dataset "$dataset" \
        --overwrite \
        --deterministic
done