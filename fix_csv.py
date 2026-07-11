from pathlib import Path

input_file = Path("evaluation/golden_dataset.csv")
output_file = Path("evaluation/golden_dataset_fixed.csv")

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

fixed = []

# Keep header
fixed.append(lines[0].strip())

for line in lines[1:]:
    line = line.strip()

    # Split only first 4 commas
    parts = line.split(",", 4)

    if len(parts) != 5:
        print("Skipping:", line)
        continue

    id_, category, difficulty, question, answer = parts

    # Remove ChatGPT citation artifacts
    answer = answer.split(":contentReference")[0].strip()

    fixed.append(
        f'{id_},{category},{difficulty},"{question}","{answer}"'
    )

with open(output_file, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(fixed))

print("Done!")
print("Saved to:", output_file)