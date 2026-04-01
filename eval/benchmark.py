# Benchmark script: measures precision@5 and end-to-end latency of the RAG pipeline.
# Usage: python -m eval.benchmark --csv path/to/dataset.csv
import time
import argparse
import pandas as pd
from rag import build_index, hybrid_search
from eval.queries import QUERIES

def precision_at_5(retrieved_chunks, expected_keywords):
    hits = 0
    for chunk in retrieved_chunks:
        chunk_lower = chunk.lower()
        if any(kw in chunk_lower for kw in expected_keywords):
            hits += 1
    return hits / len(retrieved_chunks)

def run_benchmark(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Dataset: {csv_path} | {len(df)} rows x {len(df.columns)} cols")

    t0 = time.time()
    index, bm25, chunks, _ = build_index(df)
    index_time = time.time() - t0
    print(f"Index built in {index_time:.2f}s over {len(chunks)} chunks\n")

    precisions = []
    latencies = []

    for query, expected in QUERIES:
        t_start = time.time()
        results = hybrid_search(query, index, bm25, chunks, top_k=5)
        p5 = precision_at_5(results[:5], expected)
        latency_ms = (time.time() - t_start) * 1000
        p5 = precision_at_5(results[:5], expected)
        precisions.append(p5)
        latencies.append(latency_ms)
        print(f"[{p5:.2f} P@5 | {latency_ms:.1f}ms] {query}")

    avg_p5 = sum(precisions) / len(precisions)
    avg_lat = sum(latencies) / len(latencies)
    print(f"\nAvg Precision@5: {avg_p5:.3f} | Avg Latency: {avg_lat:.1f}ms\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    run_benchmark(args.csv)