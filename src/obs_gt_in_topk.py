import argparse,os,json,torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


class MemmapPseudoDataset(Dataset):
    """Dataset backed by memmap arrays."""
    def __init__(self, data_dir, pad_token_id, limit=None):
        inputs_meta = json.load(open(os.path.join(data_dir, "inputs_int.mmp.json")))
        self.inputs = np.memmap(
            os.path.join(data_dir, "inputs_int.mmp"),
            mode="r",
            shape=tuple(inputs_meta["shape"]),
            dtype=np.uint32,
        )
        self.length = self.inputs.shape[0] if limit is None else min(limit, self.inputs.shape[0])
        self.seq_len = self.inputs.shape[1]
        self.pad_token_id = pad_token_id

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = torch.from_numpy(self.inputs[idx].astype(np.int64, copy=False))
        inp = data[:-1]
        pred = data[1:]
        return {
            "input_ids": inp,
            "labels": pred,
            "attention_mask": inp.ne(self.pad_token_id).long()
        }

def parse_args():
    p = argparse.ArgumentParser("Evaluate rank")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data_dir", type=str, help="Data directory with memmap files")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_num", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.deterministic:
        torch.manual_seed(42)
        np.random.seed(42)
        torch.use_deterministic_algorithms(True)

    topk_list = [1, 10, 100, 1000, 10000] # you may change this to your interested k values

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto").eval()
    device = model.device

    special_tokens = tokenizer.all_special_ids
    ds = MemmapPseudoDataset(args.data_dir, tokenizer.pad_token_id, limit=args.data_num)
    Len = len(ds)

    def collate(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch]).to(device)
        labels = torch.stack([x["labels"] for x in batch]).to(device)
        attention_mask = torch.stack([x["attention_mask"] for x in batch]).to(device)
        return input_ids, labels, attention_mask

    all_metrics = {}
    for tk in topk_list:
        all_metrics[tk] = {
            "results": [],
            "p_gt": [],
            "indicator": [],
        }
    global_topk_max = max(topk_list)

    for idx in tqdm(range(0, Len, args.batch_size), desc="Evaluating Batches"):
        batch = [ds[i] for i in range(idx, min(idx + args.batch_size, Len))]
        input_ids, labels, attention_mask = collate(batch)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs = torch.softmax(logits, dim=-1)
        vocab_size = probs.size(-1)
        kmax = min(global_topk_max, vocab_size)
        
        topk_kmax_probs, topk_kmax_indices = torch.topk(probs, k=kmax, dim=-1)
        target_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        rank = (logits > target_logits.unsqueeze(-1)).sum(-1) + 1

        valid_mask = attention_mask.bool() & (labels != tokenizer.pad_token_id)
        for s in special_tokens:
            valid_mask &= (input_ids != s) & (labels != s)

        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction="none"
        ).view(logits.size(0), logits.size(1))

        positions = torch.arange(1, logits.size(1) + 1, device=device).unsqueeze(0).expand_as(rank)

        for topk in topk_list:
            k = min(topk, kmax)
            topk_probs = topk_kmax_probs[..., :k]
            topk_indices = topk_kmax_indices[..., :k]

            is_correct = (labels.unsqueeze(-1) == topk_indices).any(dim=-1)
            res = all_metrics[topk]

            B, L = input_ids.size(0), input_ids.size(1)

            for i in range(B):
                mask_i = valid_mask[i]
                if mask_i.any():
                    valid_pos_idx = torch.nonzero(mask_i, as_tuple=False).squeeze(-1)
                    for kpos_idx, seq_pos in enumerate(valid_pos_idx):
                        j = int(seq_pos.item())
                        res["results"].append((
                            int(rank[i, j].cpu()),
                            int(positions[i, j].cpu()),
                            float(ce_loss[i, j].cpu())
                        ))
                        gt_prob = float(probs[i, j, labels[i, j]].cpu())
                        indicator = 1 if bool(is_correct[i, j].cpu()) else 0

                        res["p_gt"].append(gt_prob)
                        res["indicator"].append(indicator)

    # ========== save metrics ==========
    for topk in topk_list:
        res = all_metrics[topk]
        results_np = np.array(res["results"])
        all_ce = results_np[:, 2].astype(float)
        mean_ce = float(f"{all_ce.mean():.6f}")
        metrics = {
            "mean_ce_loss": mean_ce
        }
        metrics["mean_p_gt"] = float(f"{np.mean(res['p_gt']):.6f}")
        metrics[f"RBP_{topk}"] = float(f"{np.mean(res['indicator']):.6f}")
        
        topk_dir = os.path.join(args.output_dir, f"top{topk}")
        Path(topk_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(topk_dir, "ranks.json"), "w") as f:
            json.dump(res["results"], f)
        with open(os.path.join(topk_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
