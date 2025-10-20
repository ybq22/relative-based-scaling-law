import argparse,os,json,torch,math
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

    topk_list = [1, 10, 100, 1000, 10000]
    N_list = [1,5,10,50,100]

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
        attention_mask = torch.stack([x["attention_mask"] for x in batch]).to(device)
        labels = torch.stack([x["labels"] for x in batch]).to(device)
        return input_ids, labels, attention_mask

    # Store metrics for different topk values
    all_metrics = {}
    for tk in topk_list:
        all_metrics[tk] = {
            "results": [],
            "position_correct_prob_sum": defaultdict(float),
            "position_count": defaultdict(int),
            "all_correct_probs": [],
            "all_seq_lengths": [],
            # Original log-sum F statistics
            "seq_logprob_sum": {n: 0.0 for n in N_list},
            "seq_count": {n: 0 for n in N_list},
            # New perfect window statistics
            "seq_window_all_correct": {n: 0 for n in N_list},
            "seq_window_count": {n: 0 for n in N_list},
            
            "p_gt": [],
            "p_gt_in_topk": [],
            "p_gt_in_topk_indicator": [],
            "indicator": []

        }

    eps = 1e-12  # log stability
    global_topk_max = max(topk_list)

    for idx in tqdm(range(0, Len, args.batch_size), desc="Evaluating Batches"):
        batch = [ds[i] for i in range(idx, min(idx + args.batch_size, Len))]
        input_ids, labels, attention_mask = collate(batch)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs = torch.softmax(logits, dim=-1)
        vocab_size = probs.size(-1)
        kmax = min(global_topk_max, vocab_size)
        if kmax <= 0:
            raise RuntimeError("vocab_size <= 0 ?")

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

            correct_probs = topk_probs.sum(dim=-1)
            is_correct = (labels.unsqueeze(-1) == topk_indices).any(dim=-1)
            res = all_metrics[topk]

            B, L = input_ids.size(0), input_ids.size(1)

            for i in range(B):
                current_streak = 0
                mask_i = valid_mask[i]
                if mask_i.any():
                    v = correct_probs[i][mask_i]
                    valid_pos_idx = torch.nonzero(mask_i, as_tuple=False).squeeze(-1)
                    for kpos_idx, seq_pos in enumerate(valid_pos_idx):
                        j = int(seq_pos.item())
                        res["results"].append((
                            int(rank[i, j].cpu()),
                            int(positions[i, j].cpu()),
                            float(ce_loss[i, j].cpu())
                        ))
                        mp = float(correct_probs[i, j].cpu())
                        res["position_correct_prob_sum"][int(positions[i, j])] += mp
                        res["position_count"][int(positions[i, j])] += 1
                        res["all_correct_probs"].append(mp)
                        gt_prob = float(probs[i, j, labels[i, j]].cpu())  # ground truth probability
                        topk_sum_prob = float(topk_probs[i, j].sum().cpu()) + eps
                        p_gt_in_topk = gt_prob / topk_sum_prob
                        indicator = 1 if bool(is_correct[i, j].cpu()) else 0
                        indicator_val = p_gt_in_topk if bool(is_correct[i, j].cpu()) else 0.0

                        res["p_gt"].append(gt_prob)
                        res["p_gt_in_topk"].append(p_gt_in_topk)
                        res["p_gt_in_topk_indicator"].append(indicator_val)
                        res["indicator"].append(indicator)
                        if bool(is_correct[i, j].cpu()):
                            current_streak += 1
                        else:
                            if current_streak > 0:
                                res["all_seq_lengths"].append(current_streak)
                            current_streak = 0
                    if current_streak > 0:
                        res["all_seq_lengths"].append(current_streak)

                
                # ===== Original log-sum F calculation =====
                if mask_i.any():
                    log_v = torch.log(v + eps)
                    for n in N_list:
                        if v.numel() >= n:
                            window_sums = log_v.unfold(0, n, 1).sum(dim=-1)
                            res["seq_logprob_sum"][n] += float(window_sums.sum().item())
                            res["seq_count"][n] += int(window_sums.numel())

                # ===== New: Perfect window ratio F(N) =====
                if mask_i.any():
                    is_correct_i = is_correct[i][mask_i]
                    for n in N_list:
                        if is_correct_i.numel() >= n:
                            windows = is_correct_i.unfold(0, n, 1)
                            num_windows = windows.size(0)
                            num_all_correct = (windows.all(dim=-1)).sum().item()
                            res["seq_window_all_correct"][n] += num_all_correct
                            res["seq_window_count"][n] += num_windows

    # ========== Save metrics ==========
    for topk in topk_list:
        res = all_metrics[topk]
        if len(res["results"]) > 0:
            results_np = np.array(res["results"])
            all_ce = results_np[:, 2].astype(float)
            mean_ce = float(f"{all_ce.mean():.6f}")
        else:
            mean_ce = None

        if len(res["all_correct_probs"]) > 0:
            r_val = float(f"{1 - np.mean(res['all_correct_probs']):.6f}")
        else:
            r_val = None

        avg_N = float(f"{np.mean(res['all_seq_lengths']):.6f}") if res["all_seq_lengths"] else None

        metrics = {
            "mean_ce_loss": mean_ce,
            "r": r_val,
            "avg_N": avg_N
        }
        if len(res["p_gt"]) > 0:
            metrics["mean_p_gt"] = float(f"{np.mean(res['p_gt']):.6f}")
        else:
            metrics["mean_p_gt"] = None

        if len(res["p_gt_in_topk"]) > 0:
            metrics["mean_p_gt_in_topk"] = float(f"{np.mean(res['p_gt_in_topk']):.6f}")
        else:
            metrics["mean_p_gt_in_topk"] = None

        if len(res["p_gt_in_topk_indicator"]) > 0:
            metrics["mean_p_gt_in_topk_indicator"] = float(f"{np.mean(res['p_gt_in_topk_indicator']):.6f}")
        else:
            metrics["mean_p_gt_in_topk_indicator"] = None
        if len(res["indicator"]) > 0:
            metrics["indicator"] = float(f"{np.mean(res['indicator']):.6f}")


        # ===== Save original log-sum F =====
        for n in N_list:
            if res["seq_count"][n] > 0:
                avg_log_prob = res["seq_logprob_sum"][n] / res["seq_count"][n]
                metrics[f"logF_N={n}"] = float(f"{avg_log_prob:.6f}")
            else:
                metrics[f"logF_N={n}"] = None

        # ===== Save new perfect window ratio F =====
        for n in N_list:
            if res["seq_window_count"][n] > 0:
                F_n = res["seq_window_all_correct"][n] / res["seq_window_count"][n]
                metrics[f"F_N={n}"] = float(f"{F_n}")
            else:
                metrics[f"F_N={n}"] = None
        
        position_stats = {}
        for pos in sorted(res["position_count"].keys()):
            avg_max_prob = res["position_correct_prob_sum"][pos] / res["position_count"][pos]
            r_pos = 1 - avg_max_prob
            position_stats[pos] = {"r": float(f"{r_pos:.6f}")}
        metrics["position_stats"] = position_stats
        
        topk_dir = os.path.join(args.output_dir, f"top{topk}")
        Path(topk_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(topk_dir, "ranks.json"), "w") as f:
            json.dump(res["results"], f)
        with open(os.path.join(topk_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Top-{topk}: mean_ce={metrics['mean_ce_loss']} r={metrics['r']} avg_N={metrics['avg_N']}")

if __name__ == "__main__":
    main()
