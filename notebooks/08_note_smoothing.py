import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(r"D:\Musical-Score-Generation")

INPUT_FILE = (
    BASE_DIR / "outputs" / "note_segmentation"
    / "notes_0000.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "note_segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load segmented notes
df = pd.read_csv(INPUT_FILE)

df = df.sort_values("Start Time").reset_index(drop=True)

# Maximum gap allowed between identical notes
GAP_THRESHOLD = 0.12

smoothed_segments = []

for _, row in df.iterrows():

    current = {
        "Note": row["Note"],
        "Start Time": row["Start Time"],
        "End Time": row["End Time"]
    }

    if len(smoothed_segments) == 0:
        smoothed_segments.append(current)
        continue

    previous = smoothed_segments[-1]

    gap = current["Start Time"] - previous["End Time"]

    # Merge only identical notes with a small gap
    if (
        current["Note"] == previous["Note"]
        and 0 <= gap <= GAP_THRESHOLD
    ):
        previous["End Time"] = current["End Time"]

    else:
        smoothed_segments.append(current)

# Create DataFrame
result = pd.DataFrame(smoothed_segments)

# Calculate duration
result["Duration"] = (
    result["End Time"] - result["Start Time"]
)

# Save output
output_file = OUTPUT_DIR / "smoothed_notes_0000.csv"

result.to_csv(output_file, index=False)

# Display results
print("\nNOTE SMOOTHING COMPLETED!")

print("\nOriginal Segments:", len(df))
print("Smoothed Segments:", len(result))

print("\nSmoothed Musical Notes:")
print(result.to_string(index=False))

print("\nOutput saved at:", output_file)