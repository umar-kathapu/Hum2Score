import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "note_segmentation"
    / "smoothed_notes_0000.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rhythm_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "rhythm_intervals_0000.csv"
PLOT_OUTPUT = OUTPUT_DIR / "rhythm_interval_visualization_0000.png"

# Load notes
df = pd.read_csv(INPUT_FILE)

# Find onset column
possible_columns = ["Start Time", "start_time", "Start", "start"]

onset_column = next(
    (col for col in possible_columns if col in df.columns),
    None
)

if onset_column is None:
    raise ValueError(
        f"Start-time column not found. Columns: {list(df.columns)}"
    )

df[onset_column] = pd.to_numeric(
    df[onset_column],
    errors="coerce"
)

df = df.dropna(subset=[onset_column]).sort_values(onset_column).reset_index(drop=True)

if len(df) < 2:
    raise ValueError("At least two note onsets are required.")

# Calculate intervals between consecutive note onsets
df["Next Note Start"] = df[onset_column].shift(-1)

df["Onset Interval"] = (
    df["Next Note Start"] - df[onset_column]
)

intervals = df["Onset Interval"].dropna()

# Calculate statistics
mean_interval = intervals.mean()
median_interval = intervals.median()
std_interval = intervals.std()

q25 = intervals.quantile(0.25)
q75 = intervals.quantile(0.75)

print("\nRHYTHM INTERVAL ANALYSIS")
print("=" * 55)

print(f"Total Notes: {len(df)}")
print(f"Total Intervals: {len(intervals)}")

print("\nInterval Statistics:")
print(f"Mean Interval: {mean_interval:.4f} seconds")
print(f"Median Interval: {median_interval:.4f} seconds")
print(f"Standard Deviation: {std_interval:.4f} seconds")
print(f"25th Percentile: {q25:.4f} seconds")
print(f"75th Percentile: {q75:.4f} seconds")
print(f"Minimum Interval: {intervals.min():.4f} seconds")
print(f"Maximum Interval: {intervals.max():.4f} seconds")

# Classify intervals into exploratory groups
def classify_interval(value):
    if value < 0.15:
        return "Short"
    elif value < 0.30:
        return "Medium"
    else:
        return "Long"

df["Interval Category"] = df["Onset Interval"].apply(
    lambda x: classify_interval(x) if pd.notna(x) else None
)

print("\nInterval Category Counts:")
print(df["Interval Category"].value_counts(dropna=True))

# Save interval data
df.to_csv(CSV_OUTPUT, index=False)

# Plot interval sequence
plt.figure(figsize=(12, 5))

plt.plot(
    np.arange(1, len(intervals) + 1),
    intervals.to_numpy(),
    marker="o"
)

plt.axhline(
    mean_interval,
    linestyle="--",
    label=f"Mean: {mean_interval:.3f}s"
)

plt.axhline(
    median_interval,
    linestyle=":",
    label=f"Median: {median_interval:.3f}s"
)

plt.title("Consecutive Note Onset Intervals")
plt.xlabel("Note Transition Number")
plt.ylabel("Time Interval (seconds)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(PLOT_OUTPUT, dpi=300)
plt.show()

print("\nFiles Saved:")
print(CSV_OUTPUT)
print(PLOT_OUTPUT)

print("\nAnalysis Completed.")