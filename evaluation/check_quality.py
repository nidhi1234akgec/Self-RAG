import pandas as pd
import sys

results = pd.read_csv("evaluation/evaluation_results.csv")


accuracy = ((
    (results["issup"] == "fully_supported")
    &
    (results["isuse"] == "useful")
).mean()) * 100


faithfulness = (
    (results["issup"] == "fully_supported").mean()
) * 100

usefulness = (
    (results["isuse"] == "useful").mean()
) * 100

print("\n============================")
print("QUALITY GATE")
print("============================")

print(f"Accuracy      : {accuracy:.2f}%")
print(f"Faithfulness  : {faithfulness:.2f}%")
print(f"Usefulness    : {usefulness:.2f}%")

FAILED = False

if accuracy < 85:
    print("❌ Accuracy below threshold")
    FAILED = True

if faithfulness < 90:
    print("❌ Faithfulness below threshold")
    FAILED = True

if usefulness < 85:
    print("❌ Usefulness below threshold")
    FAILED = True

if FAILED:
    print("\n❌ QUALITY GATE FAILED")
    sys.exit(1)

print("\n✅ QUALITY GATE PASSED")