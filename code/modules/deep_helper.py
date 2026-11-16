"""Frozen sentence embeddings for the job postings.

sentence-transformers is only needed on the machine that builds the cache, so
the heavy import happens inside embed_texts. The module therefore compiles and
imports on any laptop.
"""

import os

import numpy as np


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

    # Cache file names are derived from the model name.
    os.makedirs(cache_dir, exist_ok=True)
    slug = model_name.replace("/", "_")
    emb_path = os.path.join(cache_dir, slug + "_emb.npy")
    ids_path = os.path.join(cache_dir, slug + "_ids.npy")

    # Reuse the cache when it was built for exactly these rows.
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        cached_ids = np.load(ids_path, allow_pickle=True)
        if np.array_equal(cached_ids, ids):
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
    return emb
