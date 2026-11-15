# Running the deep part on the GPU server (NRC)

Everything below runs from the repo root after `git pull`. The classical results and the saved splits are already in the repo
(`code/outputs/classical/`, `code/splits/`), so the server only has to produce `code/outputs/deep/`.

## 0. One-time setup
```bash
git clone https://github.com/BanAnA9205/fake-job-detection-p4ai.git && cd fake-job-detection-p4ai   # or git pull
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
pip install torch transformers sentence-transformers          # pick the CUDA build your driver supports
cd code && for t in tests/test_*.py; do python $t || break; done   # all tests should pass before any run
```

## 1. Check the GPU before launching (the server is shared)
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=<index of the emptiest GPU>   # DistilBERT, batch 16, 256 tokens, bf16 needs very roughly 4-6 GB (an estimate, not measured)
```

## 2. Run, in this order, inside tmux (it survives a dropped ssh connection)
```bash
tmux new -s deep
cd code && mkdir -p logs && export OMP_NUM_THREADS=8
# a) cheap: embeddings + the two sklearn heads, all 5 seeds (embedding 17.6k texts once; heads are CPU)
nice -n 10 python run_deep.py --stage embed --out outputs/deep 2>&1 | tee logs/deep_embed.log
nice -n 10 python run_deep.py --stage heads --out outputs/deep 2>&1 | tee logs/deep_heads.log
# b) pilot BERT on ONE seed first and read the per-epoch seconds before launching more
nice -n 10 python run_deep.py --stage bert --seeds 0 --out outputs/deep 2>&1 | tee logs/deep_bert_seed0.log
# c) if the timing is acceptable, the remaining seeds (finished runs are skipped automatically)
nice -n 10 python run_deep.py --stage bert --seeds 1 2 3 4 --out outputs/deep 2>&1 | tee logs/deep_bert_rest.log
```
If it is interrupted, rerun the same command: finished (model, seed) pairs are skipped.
Out-of-memory: lower `--batch-size` to 8 (or `--max-len` to 192). Do not use a GPU whose free memory is under ~6 GB.

## 3. Send results back
```bash
cd .. && git add code/outputs/deep && git commit -m "Deep results: embeddings heads and DistilBERT" && git push
```
Only the small json/npz result files are committed; embeddings (`code/features/*.npy`) and logs stay on the server.

## 4. Back on the laptop
`git pull`, then run `code/notebooks/03_results.ipynb` (it picks up `outputs/deep/` automatically).

## What was and was not verified before you run this
- Verified on the laptop with synthetic data and a tiny random BERT: the training loop, early stopping and best-weight restore,
  the results format, resume-skip, and `--help`.
- NOT verified (needs the real model, internet and a GPU): downloading `distilbert-base-uncased` and
  `sentence-transformers/all-MiniLM-L6-v2`, bf16 autocast on CUDA, the memory figure above, and all timings.
  The first real run is therefore the pilot; expect to fix small things.
