import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(r"D:\Musical-Score-Generation")

INPUT_FILE = (
    BASE_DIR / "outputs" / "note_segmentation"
    / "smoothed_notes_0000.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "rhythm_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load smoothed notes
df = pd.read_csv(INPUT_FILE)

df = df.sort_values("Start Time").reset_index(drop=True)

# Calculate note duration
df["Duration"] = df["End Time"] - df["Start Time"]

# Calculate gap between consecutive notes
df["Previous End Time"] = df["End Time"].shift(1)

df["Gap"] = (
    df["Start Time"] - df["Previous End Time"]
)

df["Silence Duration"] = df["Gap"].clip(lower=0)

df["Overlap Duration"] = (-df["Gap"]).clip(lower=0)

# First note has no previous gap
df.loc[0, "Gap"] = np.nan

# Calculate statistics
average_duration = df["Duration"].mean()
median_duration = df["Duration"].median()
shortest_duration = df["Duration"].min()
longest_duration = df["Duration"].max()

average_gap = df["Gap"].mean()

# Display results
print("\nRHYTHM ANALYSIS COMPLETED!")

print("\nNOTE DURATION STATISTICS")
print("--------------------------------")
print("Total Notes:", len(df))
print(f"Average Duration: {average_duration:.3f} seconds")
print(f"Median Duration: {median_duration:.3f} seconds")
print(f"Shortest Duration: {shortest_duration:.3f} seconds")
print(f"Longest Duration: {longest_duration:.3f} seconds")

print("\nINTER-NOTE GAP STATISTICS")
print("--------------------------------")
print(f"Average Gap: {average_gap:.3f} seconds")

# Save detailed results
output_file = OUTPUT_DIR / "rhythm_analysis_0000.csv"

df.to_csv(output_file, index=False)

print("\nDetailed results saved at:")
print(output_file)