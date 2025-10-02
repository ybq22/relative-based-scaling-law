from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from itertools import islice

class TextDataset(torch.utils.data.Dataset):
    """Dataset that yields text samples from a Hugging Face dataset with flexible field handling."""

    def __init__(self, ds, field_key, begin_offset: int = 0, end_offset: int = None):
        """
        Args:
            ds: Hugging Face dataset
            field_key: str 或 list[str]，需要读取的字段
            begin_offset: 起始偏移
            end_offset: 结束偏移
        """
        self.ds = ds
        self.field_key = field_key
        self.begin_offset = begin_offset
        self.end_offset = end_offset if end_offset is not None else len(ds)

    def __len__(self) -> int:
        return self.end_offset - self.begin_offset

    def __getitem__(self, idx: int) -> str:
        item = self.ds[self.begin_offset + idx]
        if isinstance(self.field_key, str):
            return item.get(self.field_key, "")
        elif isinstance(self.field_key, (list, tuple)):
            return "\n".join(str(item.get(k, "")) for k in self.field_key)
        else:
            raise ValueError(f"Unsupported field_key type: {type(self.field_key)}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Extract next‑token arg‑max predictions (batched)")
    p.add_argument("--model", default="EleutherAI/pythia-14m")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--num_sequences", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=16, help="Sequences per forward pass")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true",
                   help="Enable deterministic behavior (e.g., for reproducibility)")
    p.add_argument("--temperature", type=float, required=True,)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing memmap files if they exist")
    # 新增：数据集相关参数
    p.add_argument("--dataset_name", type=str, default="wikimedia/wikipedia")
    return p.parse_args()

def load_different_datasets(dataset_name):
    if dataset_name == "openai/gsm8k":
        data = load_dataset("openai/gsm8k", "main", split="test")
        field_key = ["question", "answer"]
    elif dataset_name == "openai/openai_humaneval":
        data = load_dataset("openai/openai_humaneval", split="test")
        field_key = ["prompt", "canonical_solution"]
    elif dataset_name == "hotpotqa/hotpot_qa":
        data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        field_key = ["question", "answer"]
    elif dataset_name == "wikimedia/wikipedia":
        data = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
        field_key = "text"
    elif dataset_name == "isaacus/open-australian-legal-corpus":
        data = load_dataset("isaacus/open-australian-legal-corpus", split='corpus')
        field_key = "text"
    elif dataset_name == "allenai/c4":
        ds_iter = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        samples = list(islice(ds_iter, 1000))
        ds = Dataset.from_list(samples)
        field_key = "text"

        return ds, field_key
    elif dataset_name == "monology/pile-uncopyrighted/Github":
        dataset_local_dir = "/data-share/guest/yuebaoqing/vsmoe/data/Github"
        data = load_dataset(dataset_local_dir, split="validation",
                            data_files={"validation": [os.path.join(dataset_local_dir, "test.jsonl.zst"),os.path.join(dataset_local_dir, "val.jsonl.zst")]})
        print(f"Loaded {len(data)} samples from {dataset_name}")
        data = data.filter(lambda x: x['meta']['pile_set_name'] == "Github" and len(x['text']) > 512)
        print(f"Loaded {len(data)} samples from Github")
        field_key = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return data, field_key

@torch.inference_mode()
def flush_batch(model, input_ids, attention_mask, temperature, special_tokens):
    input_ids = input_ids.to(model.device)  # (B, L)
    attention_mask = attention_mask.to(model.device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, L, V)
    if temperature < 1e-2: # when temperature == 0, use greedy
        preds = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.uint32, copy=False)
    else:
        heated_probs = torch.softmax(logits / temperature, dim=-1)
        preds = torch.multinomial(heated_probs.view(-1, heated_probs.size(-1)), num_samples=1).view(logits.shape[:-1]).cpu().numpy().astype(np.uint32, copy=False)
        # print("Sampled predictions (first 5 tokens):", preds[0][:5])
        # print("Top-5 probs for first token:", torch.topk(heated_probs[0,0], 5))
        
    arr_in = input_ids.cpu().numpy().astype(np.uint32, copy=False)  # (B, L)
    
    # Gather logits for targets
    target_ids = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]  # (B, L-1)
    logits = logits[:, :-1]
    target_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, L)
    # Rank = number of logits greater than target
    rank = (logits > target_logits.unsqueeze(-1)).sum(-1)  # (B, L)
    # valid tokens are those input_ids and target_ids not in special tokens
    valid_tokens = torch.ones_like(input_ids)
    for special_token in special_tokens:
        valid_tokens &= (input_ids != special_token)
        valid_tokens &= (target_ids != special_token)
    # only select ranks for attention_mask == 1
    selected_ranks = rank[valid_tokens.bool()]
    # compute cross_entropy for each token
    ce_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction='none'
    ).reshape(logits.size(0), logits.size(1))
    ce_loss = ce_loss[valid_tokens.bool()]
    # also log position_rank in the sequence
    position_rank = torch.arange(1, logits.size(1) + 1, device=logits.device).unsqueeze(0).expand(logits.size(0), -1)
    position_rank = position_rank[valid_tokens.bool()]

    # return arr_in, preds, selected_ranks.cpu().numpy().tolist(), position_rank.cpu().numpy().tolist(), ce_loss.cpu().numpy().tolist()
    # only for Qwen use
    return arr_in, preds, selected_ranks.cpu().float().numpy().tolist(), position_rank.cpu().float().numpy().tolist(), ce_loss.cpu().float().numpy().tolist()



@torch.inference_mode()
def run(output_dir, model, tokenizer, ds, field_key, seq_len, begin_offset, num_sequences, batch_size, temperature, do_overwrite=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # assert mmp does not exist
    assert do_overwrite or not (Path(output_dir) / "inputs_int.mmp").exists(), "Output memmap already exists"
    assert do_overwrite or not (Path(output_dir) / "preds_int.mmp").exists(), "Output memmap already exists"

    inputs_mm = np.memmap(Path(output_dir) / "inputs_int.mmp", mode="w+", dtype=np.uint32,
                          shape=(num_sequences, seq_len))
    # dump inputs_mm metadata
    with open(Path(output_dir) / "inputs_int.mmp.json", "w") as f:
        json.dump({
            "dtype": str(inputs_mm.dtype),
            "shape": inputs_mm.shape,
            "seq_len": seq_len,
            "num_sequences": num_sequences,
        }, f, indent=2)
    preds_mm = np.memmap(Path(output_dir) / "preds_int.mmp", mode="w+", dtype=np.uint32,
                         shape=(num_sequences, seq_len))
    # dump preds_mm metadata
    with open(Path(output_dir) / "preds_int.mmp.json", "w") as f:
        json.dump({
            "dtype": str(preds_mm.dtype),
            "shape": preds_mm.shape,
            "seq_len": seq_len,
            "num_sequences": num_sequences,
        }, f, indent=2)

    dataset = TextDataset(ds, field_key, begin_offset=begin_offset, end_offset=begin_offset + num_sequences)
    def collate_fn(batch):
        # collect text and tokenization
        enc = tokenizer(
            batch, truncation=True, max_length=seq_len, return_tensors="pt",
            padding="max_length"
        )
        return enc
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    special_tokens = tokenizer.all_special_ids
    print(f"Special token IDs: {special_tokens}, Special tokens: {[tokenizer.decode(s) for s in special_tokens]}")

    all_ranks, all_position_ranks, all_ce_losses = [], [], []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
        # batch is a dict with keys: input_ids, attention_mask
        input_ids = batch["input_ids"]
        assert (batch['attention_mask'] == (input_ids != tokenizer.pad_token_id)).all(), "Attention mask does not match input IDs"
        
        arr_in, preds, ranks, position_ranks, ce_losses = flush_batch(
            model=model,
            temperature=temperature,
            input_ids=input_ids, 
            attention_mask=batch["attention_mask"],
            special_tokens=special_tokens,
        )
        
        write_idx=i * batch_size
        inputs_mm[write_idx:write_idx + arr_in.shape[0]] = arr_in
        preds_mm[write_idx:write_idx + preds.shape[0]] = preds
        all_ranks.extend(ranks)
        all_position_ranks.extend(position_ranks)
        all_ce_losses.extend(ce_losses)

    inputs_mm.flush()
    preds_mm.flush()
    
    np.save(Path(output_dir) / "ranks.npy", np.array(all_ranks, dtype=np.int32))
    np.save(Path(output_dir) / "position_ranks.npy", np.array(all_position_ranks, dtype=np.int32))
    np.save(Path(output_dir) / "ce_losses.npy", np.array(all_ce_losses, dtype=np.float32))


def main() -> None:
    args = parse_args()

    if args.deterministic:
        # Set deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.use_deterministic_algorithms(True)

    # 1. Load & shuffle Wikipedia
    ds, field_key = load_different_datasets(args.dataset_name)
    ds = ds.shuffle(seed=args.seed)
    print(f"Loaded {len(ds)} samples from the dataset.")
    if len(ds) < args.num_sequences:
        args.num_sequences = len(ds)

    # 2. Prepare tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # device_str = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model: GPTNeoXForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
    )
    # .to(device_str)
    model.eval()

    # run 10% num_sequences for testing
    # test_num_sequences = args.num_sequences // 1000
    test_num_sequences = args.num_sequences
    # begin_offset = args.num_sequences
    begin_offset = 0
    run(
        output_dir=os.path.join(args.output_dir, "test"),
        model=model,
        tokenizer=tokenizer,
        ds=ds,
        field_key=field_key,
        seq_len=args.seq_len,
        begin_offset=begin_offset,  # Start from the end of the previous run
        num_sequences=test_num_sequences,
        batch_size=args.batch_size,
        temperature=args.temperature,
        do_overwrite=args.overwrite,
    )
    
    # run(
    #     output_dir=os.path.join(args.output_dir, "train"),
    #     model=model,
    #     tokenizer=tokenizer,
    #     ds=ds,
    #     seq_len=args.seq_len,
    #     begin_offset=0,  # Start from the beginning of the dataset
    #     num_sequences=args.num_sequences,
    #     batch_size=args.batch_size,
    #     temperature=args.temperature,
    #     do_overwrite=args.overwrite,
    # )
    


if __name__ == "__main__":
    main()
