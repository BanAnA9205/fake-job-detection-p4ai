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

## 2. Run notebook 03 (inside tmux; it survives a dropped ssh connection)
All runs happen inside the notebooks; the `.py` files in `code/modules/` are helpers only.
```bash
tmux new -s deep
cd code/notebooks && export OMP_NUM_THREADS=8
nice -n 10 ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 03_deep_learning.ipynb
```
(or open it with `jupyter lab` over an ssh tunnel and run cell by cell). The notebook embeds the texts, fits the two
heads for all 5 seeds, and fine-tunes DistilBERT for the seeds in `BERT_SEEDS`.
- **Pilot first:** `BERT_SEEDS = [0]` is the default. Read the "seconds per epoch" line, then set `BERT_SEEDS = SEEDS` and rerun;
  finished (model, seed) pairs are skipped, as are cached embeddings.
- Out-of-memory: lower `BATCH_SIZE` to 8 (or `MAX_LEN` to 192) in the constants cell. Do not use a GPU with under ~6 GB free.
- Heavy cells only run when a GPU and torch + transformers + sentence-transformers are present, so the same notebook is safe on a laptop.

## 3. Send results back
```bash
cd ../.. && git add code/outputs/deep && git commit -m "Deep results: embeddings heads and DistilBERT" && git push
```
Commit the executed notebook too if you like. Only the small json/npz result files are committed; embeddings (`code/features/*.npy`) and logs stay on the server.

## 4. Back on the laptop
`git pull`, then re-run `code/notebooks/03_deep_learning.ipynb` (with no GPU it only reads `outputs/deep/`).

## What was and was not verified before you run this
- Verified on the laptop with synthetic data and a tiny random BERT: the training loop, early stopping and best-weight restore,
  the results format, resume-skip, and `--help`.
- NOT verified (needs the real model, internet and a GPU): downloading `distilbert-base-uncased` and
  `sentence-transformers/all-MiniLM-L6-v2`, bf16 autocast on CUDA, the memory figure above, and all timings.
  The first real run is therefore the pilot; expect to fix small things.
