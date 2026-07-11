import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Read metrics
with open(ROOT / "evaluation" / "final_metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]
faithfulness = metrics["faithfulness"]
usefulness = metrics["usefulness"]

# Overall status
if (
    accuracy >= 0.85
    and faithfulness >= 0.90
    and usefulness >= 0.85
):
    status = "PASS ✅"
else:
    status = "FAIL ❌"

report = f"""# Self-RAG Evaluation Report

## Overall Status

**{status}**

---

## Summary

| Metric | Value |
|--------|-------|
| Questions Evaluated | {metrics["total_questions"]} |
| Accuracy | {accuracy:.2%} |
| Faithfulness | {faithfulness:.2%} |
| Usefulness | {usefulness:.2%} |
| Average Retries | {metrics["avg_retries"]:.2f} |
| Average Rewrites | {metrics["avg_rewrites"]:.2f} |

---

## Quality Thresholds

| Metric | Required | Result |
|---------|----------|--------|
| Accuracy | 85% | {"PASS ✅" if accuracy>=0.85 else "FAIL ❌"} |
| Faithfulness | 90% | {"PASS ✅" if faithfulness>=0.90 else "FAIL ❌"} |
| Usefulness | 85% | {"PASS ✅" if usefulness>=0.85 else "FAIL ❌"} |

---

Generated automatically by the CI Evaluation Pipeline.
"""

reports_dir = ROOT / "reports"
reports_dir.mkdir(exist_ok=True)

with open(reports_dir / "latest_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("Report generated successfully!")
print(f"Saved to {reports_dir/'latest_report.md'}")