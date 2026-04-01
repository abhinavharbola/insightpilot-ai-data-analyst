# Streamlit UI: handles file upload, session state, chat loop, and latency display.
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import InsightPilotAgent

st.set_page_config(page_title="InsightPilot", layout="wide")
st.title("InsightPilot: AI Data Analyst")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    st.info("Get your key at [Google AI Studio](https://aistudio.google.com/)")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")

if not api_key:
    st.warning("Enter your Gemini API Key to proceed.")
    st.stop()

agent = InsightPilotAgent(api_key)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
        st.sidebar.success(f"Loaded: {uploaded_file.name} ({len(df)} rows, {len(df.columns)} cols)")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    if "messages" not in st.session_state or st.session_state.get("file") != uploaded_file.name:
        st.session_state.messages = []
        st.session_state.file = uploaded_file.name
        with st.spinner("Building RAG index and analyzing dataset..."):
            insight = agent.analyze_dataset(df)
        st.session_state.messages.append({"role": "assistant", "content": insight})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg:
                st.pyplot(msg["image"])

    user_query = st.chat_input("Ask a question about your data")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...")
            try:
                code, latency_ms = agent.generate_plotting_code(df, user_query)
                with st.expander("See Python Code"):
                    st.code(code, language="python")
                fig, output = agent.execute_code(code, df)
                response_text = f"Done. (RAG retrieval + LLM: {latency_ms:.0f}ms)"
                if output:
                    response_text += f"\n\n**Output:**\n```\n{output}\n```"
                placeholder.markdown(response_text)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "image": fig
                })
            except Exception as e:
                placeholder.error(f"Error: {e}")