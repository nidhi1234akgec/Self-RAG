import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent



st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    layout="wide",
)

st.title("📊 Self-RAG Evaluation Dashboard")
st.caption("Continuous Evaluation Metrics")


history = pd.read_csv(ROOT / "metrics_history.csv")


latest = history.iloc[-1]

c1, c2, c3 = st.columns(3)

c1.metric(
    "Faithfulness",
    f"{latest['faithfulness']:.3f}"
)

c2.metric(
    "Answer Relevancy",
    f"{latest['answer_relevancy']:.3f}"
)

c3.metric(
    "Hallucination Rate",
    f"{latest['hallucination_rate']:.3f}"
)



