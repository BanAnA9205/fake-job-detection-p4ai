"""Frozen sentence embeddings for the job postings.

sentence-transformers is only needed on the machine that builds the cache, so
the heavy import happens inside embed_texts. The module therefore compiles and
imports on any laptop.
"""

import hashlib
import os

import numpy as np


def text_fingerprint(texts):
    """Short hash of the texts in order: a cache built from other text must not be reused."""
    return hashlib.sha256("\x1f".join(map(str, texts)).encode("utf-8")).hexdigest()[:16]


def embed_texts(
    texts,
    ids,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=32,
    max_seq_length=256,
    cache_dir="features",
    device="cpu",
) -> np.ndarray:
    """Return normalised sentence embeddings for texts, reusing a cache when ids and texts match.

    Embeddings are computed once per model and saved with their row ids and a fingerprint of the
    texts, so a rerun skips the expensive encoding unless the rows or their text changed.
    """
    texts = list(texts)
    ids = np.asarray(ids)
    assert len(texts) == len(ids), "texts and ids must have the same length"

    # Cache file names are derived from the model name.
    os.makedirs(cache_dir, exist_ok=True)
    slug = model_name.replace("/", "_")
    emb_path = os.path.join(cache_dir, slug + "_emb.npy")
    ids_path = os.path.join(cache_dir, slug + "_ids.npy")
    text_path = os.path.join(cache_dir, slug + "_text.txt")
    fingerprint = text_fingerprint(texts)

    # Reuse the cache when it was built for exactly these rows and texts.
    if all(os.path.exists(p) for p in (emb_path, ids_path, text_path)):
        same_ids = np.array_equal(np.load(ids_path, allow_pickle=True), ids)
        with open(text_path) as f:
            if same_ids and f.read().strip() == fingerprint:
                return np.load(emb_path)

    # Cache miss: load the encoder (imported here so laptops without it can still import this module).
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

    # Save the vectors together with their row ids for the next run.
    np.save(emb_path, emb)
    np.save(ids_path, ids)
    with open(text_path, "w") as f:
        f.write(fingerprint)
    return emb


ZERO_SHOT_PROMPT = (
    "You screen online job advertisements for recruitment fraud. "
    "Fraudulent ads are fake postings meant to collect money or personal data.\n\n"
    "Job ad:\n{ad}\n\n"
    "Is this job ad fraudulent? Answer Yes or No."
)


def zero_shot_scores(texts, model_name, device, batch_size=16, max_tokens=512):
    """P(fraud) from an instruction-tuned language model asked a Yes/No question, no training.

    The score is the model's probability of answering "Yes", renormalised over the two
    answers "Yes" and "No", read from the next-token logits (no text is generated).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Left padding keeps the last position of every row at the end of its prompt.
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()
    answer_ids = [tok.encode(a, add_special_tokens=False)[0] for a in ("Yes", "No")]

    scores = []
    for i in range(0, len(texts), batch_size):
        # Cut each ad to max_tokens tokens, then wrap it in the model's chat template.
        prompts = []
        for text in texts[i:i + batch_size]:
            ad = tok.decode(tok.encode(text, add_special_tokens=False)[:max_tokens])
            messages = [{"role": "user", "content": ZERO_SHOT_PROMPT.format(ad=ad)}]
            prompts.append(tok.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True))

        # One forward pass; keep the logits of "Yes" and "No" at the last position.
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, answer_ids].float()
        scores.append(torch.softmax(logits, dim=-1)[:, 0].cpu().numpy())
        if i % (batch_size * 50) == 0:
            print(f"  zero-shot {i + len(prompts)}/{len(texts)}", flush=True)
    return np.concatenate(scores).astype(np.float32)
