import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

st.title("📊 Self-RAG Evaluation Dashboard")

metrics_file = Path("evaluation/final_metrics.json")

if not metrics_file.exists():
    st.error("final_metrics.json not found.")
    st.stop()

with open(metrics_file) as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"] * 100
faithfulness = metrics["faithfulness"] * 100
usefulness = metrics["usefulness"] * 100
retries = metrics["avg_retries"]
rewrites = metrics["avg_rewrites"]

c1, c2, c3 = st.columns(3)

c1.metric("Accuracy", f"{accuracy:.1f}%")
c2.metric("Faithfulness", f"{faithfulness:.1f}%")
c3.metric("Usefulness", f"{usefulness:.1f}%")

st.divider()

c4, c5 = st.columns(2)

c4.metric("Average Retries", f"{retries:.2f}")
c5.metric("Average Rewrites", f"{rewrites:.2f}")

st.divider()

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=["Accuracy", "Faithfulness", "Usefulness"],
        y=[accuracy, faithfulness, usefulness],
    )
)

fig.update_layout(
    title="Evaluation Metrics",
    yaxis_title="Score (%)",
    yaxis_range=[0, 100],
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Raw Metrics")

st.json(metrics)