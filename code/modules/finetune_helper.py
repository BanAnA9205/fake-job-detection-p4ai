"""Fine-tuning a pretrained encoder for fraud detection (run by modules/deep_runs.py).

Torch and transformers are imported inside the functions, so this module can be
imported on a laptop without them.
"""

import contextlib

import numpy as np

from modules.data_helper import seed_everything

# Stop when validation PR-AUC has not improved for this many epochs.
_PATIENCE = 2


def load_model_and_tokenizer(model_name="distilbert-base-uncased", num_labels=2):
    """Load a pretrained sequence classifier and its tokenizer from the Hub or cache."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # The tokenizer and the model must come from the same checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer


def encode_texts(tokenizer, texts, max_len=256):
    """Tokenise texts in one batched call; returns CPU tensors input_ids and attention_mask."""
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    # CPU tensors keep the dataset out of GPU memory; batches move themselves later.
    return {"input_ids": enc["input_ids"].cpu(), "attention_mask": enc["attention_mask"].cpu()}


def make_loader(enc, labels, batch_size, shuffle, seed):
    """DataLoader over (input_ids, attention_mask, label); only training may shuffle."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    assert len(labels) == len(enc["input_ids"]), "labels and encodings must have the same length"
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
    # A seeded generator makes the shuffle order reproducible; val/test never shuffle.
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _amp(device):
    """bf16 autocast on cuda (roughly half the memory); a no-op context elsewhere."""
    import torch

    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def predict_proba(model, loader, device):
    """Return P(fraud) per row in loader order, computed in eval mode without gradients."""
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            # Move one batch to the device and run the forward pass.
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            with _amp(device):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            # Softmax over the two logits, keep P(fraud), back to numpy.
            probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)


def _train_epoch(model, loader, criterion, opt, sched, device, log_every = 100):
    """One pass over the training loader; returns the sample-weighted mean loss."""
    import torch

    model.train()
    total = 0.0
    for step, (input_ids, attention_mask, labels) in enumerate(loader):
        # Batch to the device.
        input_ids, attention_mask, labels = (
            input_ids.to(device), attention_mask.to(device), labels.to(device)
        )

        # Forward pass and loss.
        opt.zero_grad()  # clear gradients first, as in the course training loop
        with _amp(device):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)  # raw logits, CrossEntropyLoss applies softmax

        # Backward pass, clipped gradient step, learning-rate schedule.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # BERT grads spike early
        opt.step()
        sched.step()

        total += loss.item() * len(labels)

        # Progress line, so a long epoch does not look like a hang.
        if step % log_every == 0:
            print(f"  step {step}/{len(loader)}  loss {loss.item():.4f}", flush=True)
    return total / len(loader.dataset)


def fine_tune(
    model,
    train_loader,
    val_loader,
    y_val,
    device,
    epochs=3,
    lr=2e-5,
    weight_decay=0.01,
    class_weight=None,
    seed=0,
    log_every=100,
):
    """Fine-tune with AdamW, warm-up and decay, keeping the best validation PR-AUC weights."""
    import time

    import torch
    from sklearn.metrics import average_precision_score
    from transformers import get_linear_schedule_with_warmup

    seed_everything(seed)
    model.to(device)

    # Loss (optionally class weighted) and optimiser.
    weight = None
    if class_weight is not None:
        weight = torch.as_tensor(class_weight, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning-rate schedule: short warm-up, then linear decay to zero.
    total_steps = len(train_loader) * epochs
    # The schedule is planned for the requested epochs; early stopping just ends sooner.
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    y_val = np.asarray(y_val).astype(int)
    history, best_auc, best_state, stale = [], -np.inf, None, 0
    for epoch in range(epochs):
        # Train one epoch, then score validation PR-AUC.
        start = time.time()
        train_loss = _train_epoch(model, train_loader, criterion, opt, sched, device, log_every=log_every)
        auc = float(average_precision_score(y_val, predict_proba(model, val_loader, device)))
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_pr_auc": auc,
            "seconds": float(time.time() - start),
        })
        print(f"epoch {epoch + 1}/{epochs}: train loss {train_loss:.4f}, val PR-AUC {auc:.3f}, "
              f"{time.time() - start:.0f}s", flush=True)

        # Keep the best weights; stop after _PATIENCE epochs without improvement.
        if auc > best_auc:
            best_auc, stale = auc, 0
            # Copy to CPU so the best checkpoint does not stay on GPU memory.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= _PATIENCE:
                break

    # Restore the best epoch's weights before returning.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history
