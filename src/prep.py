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
    p.add_argument("--num_sequences", type=int, default=1_000)
    p.add_argument("--batch_size", type=int, default=16, help="Sequences per forward pass")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true",
                   help="Enable deterministic behavior (e.g., for reproducibility)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing memmap files if they exist")
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
        data = Dataset.from_list(samples)
        field_key = "text"
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
def run(output_dir, tokenizer, ds, field_key, seq_len, begin_offset, num_sequences, batch_size, do_overwrite=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # assert mmp does not exist
    assert do_overwrite or not (Path(output_dir) / "inputs_int.mmp").exists(), "Output memmap already exists"

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

    dataset = TextDataset(ds, field_key, begin_offset=begin_offset, end_offset=begin_offset + num_sequences)
    def collate_fn(batch):
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

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
        input_ids = batch["input_ids"].cpu().numpy().astype(np.uint32, copy=False)
        write_idx = i * batch_size
        inputs_mm[write_idx:write_idx + input_ids.shape[0]] = input_ids

    inputs_mm.flush()

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

    # 2. Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    test_num_sequences = args.num_sequences
    begin_offset = 0
    run(
        output_dir=os.path.join(args.output_dir, "test"),
        tokenizer=tokenizer,
        ds=ds,
        field_key=field_key,
        seq_len=args.seq_len,
        begin_offset=begin_offset,
        num_sequences=test_num_sequences,
        batch_size=args.batch_size,
        do_overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()
