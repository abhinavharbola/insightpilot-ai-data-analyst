# InsightPilot: AI-Powered Data Analyst

Natural language analytics over CSV datasets using a hybrid RAG pipeline (FAISS dense + BM25 sparse) and Google Gemini.

<p align="center">
  <img src="images/streamlit_interface.png" width="860" alt="InsightPilot Dashboard"/>
</p>

---

## How it works

`User query` → **`FAISS dense search`** + **`BM25 sparse search`** → `Reciprocal rank fusion` → `Top-5 chunks` → `Gemini prompt` → `Python code` → `Rendered chart`

Instead of passing raw dataframe context to the LLM, the RAG pipeline retrieves only the most relevant schema and statistical chunks per query, keeping prompts focused and reducing column name hallucinations.

---

## Performance

| Metric | Result |
|---|---|
| Precision@5 | 81% |
| Retrieval latency | 9.3ms avg |
| End-to-end query response | < 500ms |
| Dataset | 9,800 rows · 18 features |

---

## Stack

`Streamlit` · `Gemini 2.5 Flash` · `sentence-transformers` · `FAISS (CPU)` · `BM25` · `Pandas` · `Matplotlib` · `Seaborn`

---

## Setup

```bash
git clone https://github.com/abhinavharbola/insightpilot-ai-data-analyst
cd insightpilot-ai-data-analyst

python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

streamlit run app.py --server.fileWatcherType none
```

Get a free Gemini API key at [Google AI Studio](https://aistudio.google.com/), enter it in the sidebar, upload a CSV, and start querying.

---

## Benchmark

```bash
python -m eval.benchmark --csv your_dataset.csv
```

<p align="center">
  <img src="images/benchmarks.png" width="620" alt="Benchmark Results"/>
</p>

Update `eval/queries.py` with your dataset's actual column names before running.

---

## Structure

```
├── app.py              # Streamlit UI
├── utils.py            # RAG retrieval + LLM orchestration
├── rag/
│   ├── embedder.py     # Chunking, embedding, FAISS index
│   └── retriever.py    # Hybrid search + RRF fusion
├── eval/
│   ├── benchmark.py    # Precision@5 + latency eval
│   └── queries.py      # 18 ground-truth query-column pairs
└── requirements.txt
```
