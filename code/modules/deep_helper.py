"""Frozen sentence embeddings with content-addressed, integrity-checked caching.

Cache entries are bound to ordered (job_id, text) pairs and encoder settings.
The old model-name-only cache is intentionally NOT reused: its identity was
insufficient to detect edited text or a changed tokenizer/truncation setting.
"""

import hashlib
import json
import os
import re
import tempfile

import numpy as np

CACHE_VERSION = 2
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def _cache_identity(texts, ids, model_name, max_seq_length, model_revision):
    texts = list(texts)
    ids = np.asarray(ids)
    if len(texts) != len(ids):
        raise ValueError("texts and ids must have the same length")
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    config = {
        "schema": CACHE_VERSION,
        "model": model_name,
        "revision": model_revision,
        "max_seq_length": int(max_seq_length),
        "normalize_embeddings": True,
        "dtype": "float32",
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8"))
    for job_id, text in zip(ids, texts):
        # Length prefixes make the content hash unambiguous even for delimiter-like text.
        for value in (str(job_id), str(text)):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def cache_path(texts, ids, model_name=DEFAULT_ENCODER, max_seq_length=256,
               cache_dir="features", model_revision=None):
    """Deterministic path for this *ordered* data and encoder configuration."""
    digest = _cache_identity(texts, ids, model_name, max_seq_length, model_revision)
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", model_name).strip("_")[:75] or "encoder"
    return os.path.join(cache_dir, f"{slug}_{digest[:32]}_emb.npy")


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cached_embeddings(texts, ids, model_name=DEFAULT_ENCODER, max_seq_length=256,
                           cache_dir="features", model_revision=None):
    """Load only a complete, matching cache. Never unpickle NumPy objects."""
    texts = list(texts)
    ids = np.asarray(ids)
    path = cache_path(texts, ids, model_name, max_seq_length, cache_dir, model_revision)
    manifest_path = path + ".json"
    if not os.path.isfile(path) or not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No validated embedding cache at {path}; run the embedding step")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema") != CACHE_VERSION or manifest.get("sha256") != _file_sha256(path):
        raise ValueError("Embedding cache integrity check failed; rebuild embeddings")
    embeddings = np.load(path, allow_pickle=False)
    if (embeddings.ndim != 2 or embeddings.shape[0] != len(ids)
            or embeddings.dtype != np.float32 or not np.isfinite(embeddings).all()
            or list(embeddings.shape) != manifest.get("shape")):
        raise ValueError("Embedding cache shape/dtype/values mismatch; rebuild embeddings")
    return embeddings


def cache_exists(texts, ids, model_name=DEFAULT_ENCODER, max_seq_length=256,
                 cache_dir="features", model_revision=None):
    """Check validity, not merely file existence, without importing the encoder."""
    try:
        load_cached_embeddings(texts, ids, model_name, max_seq_length, cache_dir, model_revision)
        return True
    except (FileNotFoundError, OSError, ValueError):
        return False


def _atomic_write_cache(path, embeddings):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_data = None
    temp_manifest = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy",
                                         dir=os.path.dirname(path) or ".", delete=False) as f:
            temp_data = f.name
            np.save(f, embeddings, allow_pickle=False)
        manifest = {
            "schema": CACHE_VERSION,
            "shape": list(embeddings.shape),
            "sha256": _file_sha256(temp_data),
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8",
                                         dir=os.path.dirname(path) or ".", delete=False) as f:
            temp_manifest = f.name
            json.dump(manifest, f, sort_keys=True)
        # Manifest is committed last. Interrupted writes are never accepted as valid.
        os.replace(temp_data, path)
        os.replace(temp_manifest, path + ".json")
    finally:
        for temp in (temp_data, temp_manifest):
            if temp and os.path.exists(temp):
                os.unlink(temp)


def embed_texts(texts, ids, model_name=DEFAULT_ENCODER, batch_size=32,
                max_seq_length=256, cache_dir="features", device="cpu",
                model_revision=None) -> np.ndarray:
    """Encode ordered texts once; invalidate cache on content/config changes."""
    texts = list(texts)
    ids = np.asarray(ids)
    path = cache_path(texts, ids, model_name, max_seq_length, cache_dir, model_revision)
    try:
        return load_cached_embeddings(texts, ids, model_name, max_seq_length,
                                      cache_dir, model_revision)
    except (FileNotFoundError, OSError, ValueError):
        pass

    # Lazy import allows tests and classical ML to run without torch.
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"device": device}
    if model_revision is not None:
        model_kwargs["revision"] = model_revision
    model = SentenceTransformer(model_name, **model_kwargs)
    model.max_seq_length = max_seq_length
    embeddings = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids) or not np.isfinite(embeddings).all():
        raise ValueError("Encoder returned invalid embedding dimensions or values")
    _atomic_write_cache(path, embeddings)
    return embeddings
