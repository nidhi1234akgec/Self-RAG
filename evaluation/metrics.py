import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Read evaluation results
results = pd.read_csv(ROOT / "evaluation_results.csv")

total = len(results)

# -----------------------------
# Compute Metrics
# -----------------------------

faithfulness = (
    results["issup"] == "fully_supported"
).mean()

usefulness = (
    results["isuse"] == "useful"
).mean()

avg_retries = results["retries"].mean()

avg_rewrites = results["rewrite_tries"].mean()

# Simple accuracy approximation
accuracy = (
    (results["issup"] == "fully_supported")
    &
    (results["isuse"] == "useful")
).mean()

metrics = {
    "total_questions": int(total),
    "accuracy": round(float(accuracy), 4),
    "faithfulness": round(float(faithfulness), 4),
    "usefulness": round(float(usefulness), 4),
    "avg_retries": round(float(avg_retries), 2),
    "avg_rewrites": round(float(avg_rewrites), 2),
}

# Save JSON
with open(ROOT / "final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nEvaluation Metrics")
print("-" * 40)

for k, v in metrics.items():
    print(f"{k:20}: {v}")

print("\nSaved to evaluation/final_metrics.json")