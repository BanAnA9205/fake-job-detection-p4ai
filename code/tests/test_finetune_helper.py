"""Smoke tests for modules/finetune_helper.py on a tiny randomly initialised BERT.

No pretrained weights are downloaded and no real data is used: the tokenizer
gets a hand-written 14-word vocabulary and the model is a 2-layer BERT built
from BertConfig. Everything runs on CPU with 32 synthetic rows.

Run: .venv/bin/python tests/test_finetune_helper.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from modules.finetune_helper import (
    encode_texts,
    fine_tune,
    make_loader,
    predict_proba,
)

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "fraud", "wire", "money", "quick",
         "job", "boring", "team", "please", "apply"]
MAX_LEN = 16

_TMP = tempfile.TemporaryDirectory(prefix="tiny_bert_")


class RecordingModel(BertForSequenceClassification):
    """BERT that saves a copy of its weights every time eval() is called."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshots = []

    def eval(self):
        # fine_tune evaluates after every training epoch, so snapshot i is "after epoch i+1".
        self.snapshots.append({k: v.detach().clone() for k, v in self.state_dict().items()})
        return super().eval()



def make_tokenizer():
    """Tokenizer over the tiny hand-made vocabulary, no download needed."""
    vocab_file = os.path.join(_TMP.name, "vocab.txt")
    if not os.path.exists(vocab_file):
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write("\n".join(VOCAB) + "\n")
    try:
        # transformers 5 takes the vocab as the first argument; transformers 4 spells it vocab_file.
        return BertTokenizerFast(vocab=vocab_file, do_lower_case=True)
    except TypeError:
        return BertTokenizerFast(vocab_file=vocab_file, do_lower_case=True)


def make_model():
    """Randomly initialised 2-layer BERT classifier; max positions cover MAX_LEN."""
    config = BertConfig(
        vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
        num_labels=2,
    )
    return RecordingModel(config)


def keyword_rows(n=32):
    """Every second row carries fraud words and label 1, the others are neutral (label 0)."""
    texts, labels = [], []
    for i in range(n):
        if i % 2 == 0:
            texts.append("please wire money quick fraud")
            labels.append(1)
        else:
            texts.append("boring team job please apply")
            labels.append(0)
    return texts, np.array(labels, dtype=int)




def test_encode_texts_shapes():
    """Tokenisation pads and truncates to exactly max_len with a matching mask."""
    tok = make_tokenizer()
    enc = encode_texts(tok, ["please wire money", "boring job"], max_len=8)
    assert set(enc) == {"input_ids", "attention_mask"}
    assert enc["input_ids"].shape == (2, 8)
    assert enc["attention_mask"].shape == (2, 8)
    assert int(enc["input_ids"].max()) < len(VOCAB), "token id outside the tiny vocab"
    assert int(enc["attention_mask"].sum()) < 2 * 8, "short rows must be padded"
    long_enc = encode_texts(tok, [" ".join(["money"] * 50)], max_len=8)
    assert long_enc["input_ids"].shape == (1, 8), "long rows must be truncated"
    assert int(long_enc["attention_mask"].sum()) == 8


def test_make_loader_order():
    """Same seed gives the same shuffle; shuffle=False keeps the input order."""
    tok = make_tokenizer()
    texts = [" ".join(["money"] * (i + 1)) + " job" for i in range(6)]
    labels = np.array([0, 1, 0, 1, 0, 1])
    enc = encode_texts(tok, texts, max_len=MAX_LEN)

    def order(loader):
        return torch.cat([batch[0] for batch in loader])

    a = make_loader(enc, labels, batch_size=4, shuffle=True, seed=0)
    b = make_loader(enc, labels, batch_size=4, shuffle=True, seed=0)
    assert torch.equal(order(a), order(b)), "same seed must give the same order"
    plain = make_loader(enc, labels, batch_size=4, shuffle=False, seed=0)
    assert torch.equal(order(plain), enc["input_ids"]), "no shuffle means input order"
    assert all(batch[2].dtype == torch.long for batch in a), "labels must be long tensors"


def test_fine_tune_and_predict():
    """Two epochs run, the history has the agreed keys, and the best weights come back."""
    torch.manual_seed(0)
    tok = make_tokenizer()
    texts, labels = keyword_rows()
    enc = encode_texts(tok, texts, max_len=MAX_LEN)
    train_enc = {k: v[:24] for k, v in enc.items()}
    val_enc = {k: v[24:] for k, v in enc.items()}
    y_tr, y_va = labels[:24], labels[24:]

    train_loader = make_loader(train_enc, y_tr, batch_size=8, shuffle=True, seed=0)
    val_loader = make_loader(val_enc, y_va, batch_size=8, shuffle=False, seed=0)
    model = make_model()
    before = [p.detach().clone() for p in model.parameters()]

    model, history = fine_tune(
        model, train_loader, val_loader, y_va, "cpu", epochs=2, lr=1e-3,
        class_weight=np.array([1.0, 2.0]), seed=0,
    )
    snapshots = list(model.snapshots)  # copy now: predict_proba below adds another snapshot

    assert 1 <= len(history) <= 2, "one entry per epoch that actually ran"
    for entry in history:
        assert set(entry) == {"epoch", "train_loss", "val_pr_auc", "seconds"}
        assert np.isfinite(entry["train_loss"]) and entry["train_loss"] > 0
        assert 0.0 <= entry["val_pr_auc"] <= 1.0
    assert [e["epoch"] for e in history] == list(range(1, len(history) + 1))
    assert model.training is False, "fine_tune must leave the model in eval mode"
    assert any(not torch.equal(p, q) for p, q in zip(before, model.parameters())), (
        "training changed no weights"
    )

    score = predict_proba(model, val_loader, "cpu")
    assert isinstance(score, np.ndarray) and len(score) == len(y_va)
    assert float(score.min()) >= 0.0 and float(score.max()) <= 1.0, "probabilities must be in [0, 1]"

    # fine_tune must reload the best epoch, not keep the last one. With PR-AUC capped
    # at 1.0 the best epoch is the first, so the last epoch is the wrong one here.
    best = int(np.argmax([e["val_pr_auc"] for e in history]))
    assert len(snapshots) == len(history) + 1, "one snapshot per eval plus the final eval"
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), snapshots[best][key]), (
            f"weights of epoch {best + 1} were not restored"
        )


if __name__ == "__main__":
    test_encode_texts_shapes()
    test_make_loader_order()
    test_fine_tune_and_predict()
    print("all finetune_helper smoke tests passed")
