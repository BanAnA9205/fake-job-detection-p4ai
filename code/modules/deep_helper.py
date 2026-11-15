"""Optional deep helpers: frozen sentence embeddings and a small MLP head.

Torch, transformers and sentence-transformers are missing from the course
environment, so every heavy import happens inside a function. The module can
therefore be compiled and imported without them.
"""

import copy
import os

import numpy as np
from sklearn.metrics import average_precision_score

from modules.data_helper import seed_everything

# Stop training when validation PR-AUC has not improved for this many epochs.
_PATIENCE = 5

fine_tune_note = """How a full BERT fine-tune would be done (Fraud-BERT setting), not implemented here.

Replace the frozen sentence embeddings with a pretrained encoder such as
bert-base-uncased and train all weights with transformers:
AutoTokenizer(max_length=256, truncation=True, padding="max_length") on the
joined text columns, AutoModelForSequenceClassification(num_labels=2), AdamW
with lr 2e-5, batch size 16 and 2 to 3 epochs. Use class weights (or a weighted
CrossEntropyLoss) for the 4.8 percent positive rate, score validation PR-AUC
after every epoch, keep the best checkpoint, and touch the test set once.
This is skipped because torch and transformers are not installed in the course
environment; only embed_texts plus the MLP head are provided.
"""


def embed_texts(
    texts,
    ids,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=32,
    max_seq_length=256,
    cache_dir="features",
    device="cpu",
) -> np.ndarray:
    """Return normalised sentence embeddings for texts, reusing a cache when the ids match.

    Embeddings are computed once per model and saved with their row ids, so a
    rerun after an unchanged split can skip the expensive encoding step.
    """
    texts = list(texts)
    ids = np.asarray(ids)
    assert len(texts) == len(ids), "texts and ids must have the same length"

    os.makedirs(cache_dir, exist_ok=True)
    slug = model_name.replace("/", "_")
    emb_path = os.path.join(cache_dir, slug + "_emb.npy")
    ids_path = os.path.join(cache_dir, slug + "_ids.npy")

    if os.path.exists(emb_path) and os.path.exists(ids_path):
        cached_ids = np.load(ids_path, allow_pickle=True)
        if np.array_equal(cached_ids, ids):
            return np.load(emb_path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    # encode keeps the input order, so row i matches ids[i]; we never shuffle.
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    assert emb.shape[0] == len(ids), "embedding row count mismatch"

    np.save(emb_path, emb)
    np.save(ids_path, ids)
    return emb


def make_mlp(in_dim, hidden=128, dropout=0.3):
    """Build the MLP head: Linear -> ReLU -> Dropout -> Linear with one logit."""
    from torch import nn

    # The class lives inside the factory because nn.Module must be imported lazily.
    class MLPHead(nn.Module):
        """Small feed-forward head on top of frozen embeddings."""

        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            """Return one raw logit per row."""
            return self.layers(x).squeeze(1)

    return MLPHead()


def _train_epoch(model, loader, criterion, opt, device):
    """Run one pass over the loader and return the sample-weighted mean loss."""
    model.train()
    total = 0.0
    for xb, yb in loader:
        opt.zero_grad()  # clear gradients first so no stale values accumulate
        loss = criterion(model(xb.to(device)), yb.to(device))
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


def _val_pr_auc(model, X_t, y):
    """Return validation PR-AUC from sigmoid probabilities, computed without gradients."""
    import torch

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).cpu().numpy()
    return float(average_precision_score(y, probs))


def train_mlp(
    Xtr,
    ytr,
    Xva,
    yva,
    epochs=20,
    lr=1e-3,
    batch_size=64,
    pos_weight=None,
    seed=0,
    device="cpu",
):
    """Train the MLP head with early stopping on validation PR-AUC; returns (model, history)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(seed)
    Xtr_t = torch.as_tensor(np.asarray(Xtr, dtype=np.float32))
    ytr_t = torch.as_tensor(np.asarray(ytr, dtype=np.float32))
    Xva_t = torch.as_tensor(np.asarray(Xva, dtype=np.float32), device=device)

    model = make_mlp(Xtr_t.shape[1]).to(device)
    weight = None
    if pos_weight is not None:
        weight = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    history = []
    best_auc, best_state, stale = -np.inf, None, 0
    for epoch in range(epochs):
        train_loss = _train_epoch(model, loader, criterion, opt, device)
        auc = _val_pr_auc(model, Xva_t, np.asarray(yva))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_pr_auc": auc})
        if auc > best_auc:
            best_auc, stale = auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= _PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_scores(model, X, device="cpu") -> np.ndarray:
    """Return fraud probabilities (sigmoid of the logits) for the rows of X."""
    import torch

    X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).cpu().numpy()
    assert probs.shape[0] == X_t.shape[0], "one score per row expected"
    return probs.astype(np.float64)
