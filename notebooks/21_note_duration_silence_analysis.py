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

CSV_OUTPUT = OUTPUT_DIR / "note_duration_silence_0000.csv"
PLOT_OUTPUT = OUTPUT_DIR / "note_duration_silence_0000.png"

# Load note data
df = pd.read_csv(INPUT_FILE)

# Identify columns
column_map = {
    "start": ["Start Time", "start_time", "Start", "start"],
    "end": ["End Time", "end_time", "End", "end"],
    "note": ["Note", "note", "Note Name", "Note_Name"]
}

def find_column(candidates):
    return next(
        (col for col in candidates if col in df.columns),
        None
    )

start_col = find_column(column_map["start"])
end_col = find_column(column_map["end"])
note_col = find_column(column_map["note"])

if start_col is None or end_col is None:
    raise ValueError(
        f"Required columns not found. Available columns: {list(df.columns)}"
    )

# Convert timestamps
df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
df[end_col] = pd.to_numeric(df[end_col], errors="coerce")

df = df.dropna(subset=[start_col, end_col])
df = df.sort_values(start_col).reset_index(drop=True)

if len(df) < 2:
    raise ValueError("At least two valid note segments are required.")

# Calculate note duration
df["Note Duration"] = df[end_col] - df[start_col]

# Calculate next note onset
df["Next Note Start"] = df[start_col].shift(-1)

# Calculate gap between current note end and next note start
df["Gap"] = df["Next Note Start"] - df[end_col]

# Separate silence and overlap
df["Silence Duration"] = df["Gap"].clip(lower=0)
df["Overlap Duration"] = (-df["Gap"]).clip(lower=0)

# Classify transition
df["Transition Type"] = np.select(
    [
        df["Gap"] > 0.01,
        df["Gap"] < -0.01
    ],
    [
        "Silence",
        "Overlap"
    ],
    default="Continuous"
)

# Remove final row for transition statistics
transitions = df.dropna(subset=["Next Note Start"]).copy()

print("\nNOTE DURATION AND SILENCE ANALYSIS")
print("=" * 60)

print(f"Total Notes: {len(df)}")
print(f"Total Transitions: {len(transitions)}")

print("\nNote Duration Statistics:")
print(f"Mean: {df['Note Duration'].mean():.4f} sec")
print(f"Median: {df['Note Duration'].median():.4f} sec")
print(f"Minimum: {df['Note Duration'].min():.4f} sec")
print(f"Maximum: {df['Note Duration'].max():.4f} sec")

print("\nTransition Statistics:")
print(f"Silence Transitions: {(transitions['Transition Type'] == 'Silence').sum()}")
print(f"Continuous Transitions: {(transitions['Transition Type'] == 'Continuous').sum()}")
print(f"Overlap Transitions: {(transitions['Transition Type'] == 'Overlap').sum()}")

print("\nSilence Statistics:")
print(f"Total Silence: {transitions['Silence Duration'].sum():.4f} sec")

if (transitions["Silence Duration"] > 0).any():
    print(
        "Average Silence: "
        f"{transitions.loc[transitions['Silence Duration'] > 0, 'Silence Duration'].mean():.4f} sec"
    )
else:
    print("Average Silence: 0 sec")

print("\nOverlap Statistics:")
print(f"Total Overlap: {transitions['Overlap Duration'].sum():.4f} sec")

# Save detailed analysis
df.to_csv(CSV_OUTPUT, index=False)

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(13, 9))

# Note durations
axes[0].plot(
    np.arange(1, len(df) + 1),
    df["Note Duration"],
    marker="o"
)

axes[0].set_title("Individual Note Durations")
axes[0].set_xlabel("Note Number")
axes[0].set_ylabel("Duration (seconds)")
axes[0].grid(True, alpha=0.3)

# Silence and overlap
x = np.arange(1, len(transitions) + 1)

axes[1].bar(
    x,
    transitions["Silence Duration"],
    label="Silence Duration"
)

axes[1].bar(
    x,
    -transitions["Overlap Duration"],
    label="Overlap Duration"
)

axes[1].axhline(0, linewidth=0.8)
axes[1].set_title("Silence and Overlap Between Notes")
axes[1].set_xlabel("Note Transition Number")
axes[1].set_ylabel("Duration (seconds)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_OUTPUT, dpi=300)
plt.show()

print("\nFiles Saved:")
print(CSV_OUTPUT)
print(PLOT_OUTPUT)

print("\nAnalysis Completed.")