# Relative-Based Scaling Law for Neural Language Models

> Anonymous authors — ICLR 2026 submission  
> **Note:** This repository is anonymized for double-blind review. No author or affiliation information is included.

---

## Overview

This repository provides code for the paper *Relative-Based Scaling Law for Neural Language Models*.  




## Data Preparation

⚠️ **Important:**  Some datasets must be downloaded **manually** before running experiments. 
Please place them under a **local directory** and specify the path in [`src/prep.py`](src/prep.py). For example: 

- **Pile-Uncopyrighted** (`monology/pile-uncopyrighted`):
  Requires `val.jsonl.zst` and `test.jsonl.zst` under `pile-uncopyrighted` 

Other datasets are loaded directly from HuggingFace Hub.  




## Repository Layout

```
.
├── prep.sh          # Prepare Tokenized Dataset
├── obs_gt_in_topk.sh          # Main Bash script: recording RBP_k
├── obs_N.sh          # Explain emergence, Section 5.1
├── plot.py        # Visualization for all
├── rank_dist.py        # Draw long-tail distribution of ranking, Section 5.2
├── requirements.txt        # Python dependencies
│
├── assets/             
│   └── param_count.json         # Non-embedding parameter counts for models used in our experiment
│
└── src/
    ├── prep.py          # Load datasets and conduct tokenization
    ├── obs_gt_in_topk.py     # Main python script
    └── obs_N.py     # Explain emergence, Section 5.1, python script
```



## Quick Start

1. **Environmental Setup**

   `pip install -r requirements.txt`

2. **Prepare Tokenized Dataset**

   `./prep.sh`

   You can specify your models and datasets in this bash script.

   If you want to add more datasets other than mentioned in our paper, please modify the `load_different_datasets` function in [`src/prep.py`](src/prep.py)

3. **Inference** 

   `./obs_gt_in_topk.sh `

   You can specify your models and datasets in this bash script. Default topk_list is 

   `[1, 10, 100, 1000, 10000]`, you can change to your own in [`src/obs_gt_in_topk.py`](src/obs_gt_in_topk.py)
   

4. **Visualization**

   Run `python plot.py` , and the generated plots will be stored in [`assets`](assets) folder.

5. **Application in Section 5**

   Run `./obs_N.sh` and `python plot.py` to observe emergence phenomenon in Section 5.1.
   
   Run `python hypo.py` to validate our hypothesis towards a unified explanation of scaling laws.





## Citation (anonymous for review)

```bibtex
@inproceedings{anonymous2026rbpscaling,
  title     = {Relative-Based Scaling Law for Neural Language Models},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```