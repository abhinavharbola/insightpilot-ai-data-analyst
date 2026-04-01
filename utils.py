# InsightPilotAgent: uses hybrid RAG retrieval to ground LLM prompts before code generation.
import io
import contextlib
import time
import google.generativeai as genai
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rag import build_index, hybrid_search

class InsightPilotAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self._index = None
        self._bm25 = None
        self._chunks = None

    def index_dataframe(self, df: pd.DataFrame):
        self._index, self._bm25, self._chunks, _ = build_index(df)

    def _retrieve(self, query: str, top_k=5) -> str:
        if self._index is None:
            return ""
        results = hybrid_search(query, self._index, self._bm25, self._chunks, top_k)
        return "\n".join(results)

    def analyze_dataset(self, df: pd.DataFrame) -> str:
        self.index_dataframe(df)
        context = self._retrieve("dataset overview column statistics", top_k=8)
        prompt = f"""You are a Senior Data Scientist. Dataset context retrieved via RAG:

{context}

Suggest 3 interesting analytical questions for this dataset. Return a clean bulleted list."""
        return self.model.generate_content(prompt).text

    def generate_plotting_code(self, df: pd.DataFrame, user_query: str):
        start = time.time()
        context = self._retrieve(user_query, top_k=5)
        exact_columns = ", ".join([f'"{c}"' for c in df.columns])
        prompt = f"""You are a Python Data Visualization Expert.

Exact column names in the dataframe (use these verbatim):
{exact_columns}

Relevant dataset context (retrieved):
{context}

User request: "{user_query}"

Write executable Python code using pandas + matplotlib/seaborn.
Rules:
- Dataframe is already in variable `df`.
- Use plt.figure() to start. Do NOT call plt.show().
- Always use the exact column names listed above. Do not guess or modify them.
- Always create figures with `fig, ax = plt.subplots()` and use `ax` for all plotting calls instead of `plt` directly.
- In seaborn calls, never pass `palette` without also setting `hue`. Instead use `color` for single-series plots.
- Return ONLY raw Python code. No markdown, no backticks."""
        response = self.model.generate_content(prompt)
        latency_ms = (time.time() - start) * 1000
        code = response.text.replace("```python", "").replace("```", "").strip()
        return code, latency_ms

    def execute_code(self, code: str, df: pd.DataFrame):
        local_vars = {"df": df, "pd": pd, "plt": plt, "io": io}
        stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, {}, local_vars)
            fig = plt.gcf()
            if not fig.get_axes():
                return None, "No plot was generated."
            return fig, stdout_capture.getvalue()
        except Exception as e:
            return None, f"Error executing code: {e}"