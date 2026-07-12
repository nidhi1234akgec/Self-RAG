import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import time
import pandas as pd

from rag.graph import ask



# =====================================================
# Load only first 20 evaluation questions
# =====================================================

dataset = pd.read_csv(
    ROOT / "evaluation" / "golden_dataset.csv"
).head(100)


# =====================================================
# Evaluation Settings
# =====================================================

MAX_RETRIES = 5
INITIAL_WAIT = 15      # seconds


results = []

print("=" * 70)
print("Starting Self-RAG Evaluation")
print("=" * 70)
print(f"Evaluating {len(dataset)} questions...\n")


# =====================================================
# Evaluation Loop
# =====================================================

for index, row in dataset.iterrows():

    question = row["question"]
    expected = row["expected_answer"]

    print(f"[{index+1}/{len(dataset)}] {question}")

    wait_time = INITIAL_WAIT

    for attempt in range(MAX_RETRIES):

        try:

            response = ask(question)

            results.append(
                {
                    "question": question,
                    "expected_answer": expected,
                    "generated_answer": response["answer"],
                    "context": response.get("context", ""),
                    "issup": response.get("issup", ""),
                    "isuse": response.get("isuse", ""),
                    "retries": response.get("retries", 0),
                    "rewrite_tries": response.get("rewrite_tries", 0),
                }
            )

            # Small pause so we don't spam the API
            time.sleep(2)

            break

        except Exception as e:

            error = str(e)

            if "429" in error or "rate limit" in error.lower():

                print(
                    f"   Rate limit hit."
                    f" Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

                wait_time *= 2

                continue

            print(f"   Error: {e}")

            results.append(
                {
                    "question": question,
                    "expected_answer": expected,
                    "generated_answer": "ERROR",
                    "context": "",
                    "issup": "",
                    "isuse": "",
                    "retries": "",
                    "rewrite_tries": "",
                }
            )

            break

    else:

        print("   Skipped after repeated rate limits.")

        results.append(
            {
                "question": question,
                "expected_answer": expected,
                "generated_answer": "RATE_LIMIT",
                "context": "",
                "issup": "",
                "isuse": "",
                "retries": "",
                "rewrite_tries": "",
            }
        )


# =====================================================
# Save Results
# =====================================================

results_df = pd.DataFrame(results)

output_path = ROOT / "evaluation" / "evaluation_results.csv"

results_df.to_csv(
    output_path,
    index=False,
)


print("\n" + "=" * 70)
print("Evaluation Completed Successfully")
print("=" * 70)

print(f"Questions Evaluated : {len(results_df)}")
print(f"Results Saved To    : {output_path}")