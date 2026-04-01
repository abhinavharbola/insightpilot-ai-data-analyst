import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .embedder import MODEL, EMBED_DIM, _to_512

def _rrf(ranks_list, k=60):
    scores = {}
    for ranks in ranks_list:
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def hybrid_search(query: str, index, bm25, chunks, top_k=5):
    raw = MODEL.encode([query], show_progress_bar=False)
    q_vec = _to_512(raw)
    faiss.normalize_L2(q_vec)

    _, dense_idxs = index.search(q_vec, top_k * 2)
    dense_ranks = dense_idxs[0].tolist()

    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    sparse_ranks = np.argsort(bm25_scores)[::-1][: top_k * 2].tolist()

    fused = _rrf([dense_ranks, sparse_ranks])[:top_k]
    return [chunks[i] for i in fused]