"""Discover candidate intents by clustering customer messages.

UMAP -> HDBSCAN rather than k-means, for two reasons that matter here:
  * k-means forces every message into a cluster. Support inboxes contain genuine
    one-offs, and pretending they belong somewhere manufactures a tidy taxonomy
    that the data does not support. HDBSCAN labels them noise (-1), which is an
    honest answer and becomes the `other` class.
  * k needs choosing up front. Density clustering lets the shape of the data
    argue for the number of intents instead.

The output is a CANDIDATE taxonomy. Clusters are geometry, not intents; a human
still has to merge, split and name them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterResult:
    labels: np.ndarray          # -1 = noise
    reduced: np.ndarray
    n_clusters: int
    noise_frac: float
    seed: int

    def sizes(self) -> dict[int, int]:
        u, c = np.unique(self.labels[self.labels >= 0], return_counts=True)
        return dict(sorted(zip(u.tolist(), c.tolist()), key=lambda kv: -kv[1]))


def reduce_dims(vecs: np.ndarray, n_components: int = 15, seed: int = 0,
                n_neighbors: int = 30) -> np.ndarray:
    import umap
    r = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                  min_dist=0.0, metric="cosine", random_state=seed)
    return r.fit_transform(vecs)


def cluster(vecs: np.ndarray, min_cluster_size: int = 100, seed: int = 0,
            reduced: np.ndarray | None = None) -> ClusterResult:
    import hdbscan
    red = reduce_dims(vecs, seed=seed) if reduced is None else reduced
    h = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                        min_samples=10, metric="euclidean",
                        cluster_selection_method="eom")
    labels = h.fit_predict(red)
    n = int(labels.max()) + 1 if labels.max() >= 0 else 0
    return ClusterResult(labels=labels, reduced=red, n_clusters=n,
                         noise_frac=float((labels < 0).mean()), seed=seed)


def exemplars(texts: list[str], vecs: np.ndarray, labels: np.ndarray,
              cid: int, k: int = 12, seed: int = 0,
              centroid_frac: float = 0.5) -> list[str]:
    """A mix of centroid-nearest and randomly drawn cluster members.

    Centroid-nearest alone is actively misleading on diffuse clusters. The most
    central message in a grab-bag is by construction its most generic one, so
    the exemplars read as "I need help" while the cluster is really a long tail
    of specific, unrelated problems. That artefact caused both a human and an
    LLM reviewer to mislabel this corpus's largest cluster (16.8% of all
    messages) as vague when it is nothing of the sort. Half the sample is now
    drawn uniformly so the true spread is visible.
    """
    idx = np.flatnonzero(labels == cid)
    if len(idx) == 0:
        return []
    centroid = vecs[idx].mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-9)
    order = idx[np.argsort(-(vecs[idx] @ centroid))]

    n_c = min(int(round(k * centroid_frac)), len(order))
    near = order[:n_c].tolist()
    rest = [i for i in idx.tolist() if i not in set(near)]
    rng = np.random.default_rng(seed + cid)
    n_r = min(k - n_c, len(rest))
    rand = rng.choice(rest, size=n_r, replace=False).tolist() if n_r else []
    return [texts[i] for i in near + rand]
