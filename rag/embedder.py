import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 512  # all-MiniLM-L6-v2 outputs 384; we pad to 512 for index consistency

def _to_512(vecs: np.ndarray) -> np.ndarray:
    if vecs.shape[1] < EMBED_DIM:
        pad = np.zeros((vecs.shape[0], EMBED_DIM - vecs.shape[1]), dtype=np.float32)
        vecs = np.hstack([vecs, pad])
    return vecs.astype(np.float32)

def build_index(df: pd.DataFrame):
    schema_chunk = "Columns: " + ", ".join(
        f"{col}({dtype})" for col, dtype in zip(df.columns, df.dtypes)
    )
    stat_chunks = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            stat_chunks.append(
                f"{col}: min={s.min():.2f} max={s.max():.2f} mean={s.mean():.2f} std={s.std():.2f}"
            )
        else:
            top = s.value_counts().head(5).index.tolist()
            stat_chunks.append(f"{col}: top values = {top}")

    sample_chunks = [
        "Row sample: " + row.to_json()
        for _, row in df.sample(min(200, len(df)), random_state=42).iterrows()
    ]

    all_chunks = [schema_chunk] + stat_chunks + sample_chunks

    raw_vecs = MODEL.encode(all_chunks, batch_size=64, show_progress_bar=False)
    vecs = _to_512(raw_vecs)
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vecs)

    bm25_corpus = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(bm25_corpus)

    return index, bm25, all_chunks, vecs