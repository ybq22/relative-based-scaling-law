# Relative-Based Scaling Law for Neural Language Models



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

   You can reproduce our results in Section 5.2 of our paper by running `python rank_dist.py`



**Now, choose your favourite model series, datasets and topk values and begin your exploration of relative-based scaling law!**



## Appendix

We are sorry to say that the derivation of origns of two scaling laws have some issues and we have updated the proof in a new version of our paper: [new-version.pdf](new-version.pdf). Several other minor adjustments have also been made in this version of paper. Thank you for understanding.
