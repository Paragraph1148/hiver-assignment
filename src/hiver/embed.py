"""Local sentence embeddings, cached to disk.

Runs on CPU via all-MiniLM-L6-v2. Deliberately local rather than an embedding
API: it costs nothing, has no rate limit, and keeps the whole retrieval and
clustering path runnable by a grader with no credentials at all. Only
generation and judging need a provider.

Embeddings are cached as .npy keyed by (model, text-list hash) so re-runs are
instant and reproducible.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path("data/interim/emb")


def _key(texts: list[str], model: str) -> str:
    h = hashlib.sha256(model.encode())
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embed(texts: list[str], model: str = DEFAULT_MODEL,
          cache_dir: Path = CACHE_DIR, batch_size: int = 256,
          show_progress: bool = True) -> np.ndarray:
    """L2-normalised embeddings, so a dot product is cosine similarity."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_key(texts, model)}.npy"
    if path.exists():
        return np.load(path)

    from sentence_transformers import SentenceTransformer  # heavy; import lazily

    m = SentenceTransformer(model, device="cpu")
    vecs = m.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=show_progress)
    vecs = vecs.astype(np.float32)
    np.save(path, vecs)
    return vecs
