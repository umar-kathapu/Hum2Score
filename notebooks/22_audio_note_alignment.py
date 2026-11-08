import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_FILE = PROJECT_ROOT / "processed_data" / "hum" / "0000.wav"

NOTES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "note_segmentation"
    / "smoothed_notes_0000.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pitch_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "audio_note_alignment_0000.png"

# Load audio
y, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)

duration = len(y) / sr
time_axis = np.arange(len(y)) / sr

# Load detected notes
df = pd.read_csv(NOTES_FILE)

# Identify required columns
def find_column(candidates):
    return next(
        (col for col in candidates if col in df.columns),
        None
    )

start_col = find_column(
    ["Start Time", "start_time", "Start", "start"]
)

end_col = find_column(
    ["End Time", "end_time", "End", "end"]
)

note_col = find_column(
    ["Note", "note", "Note Name", "Note_Name"]
)

if start_col is None or end_col is None or note_col is None:
    raise ValueError(
        f"Required columns missing. Available columns: {list(df.columns)}"
    )

df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
df[end_col] = pd.to_numeric(df[end_col], errors="coerce")

df = df.dropna(subset=[start_col, end_col])
df = df.sort_values(start_col).reset_index(drop=True)

# Check note boundaries
if (df[end_col] <= df[start_col]).any():
    raise ValueError("Invalid note duration detected.")

if (df[start_col] < 0).any() or (df[end_col] > duration + 0.1).any():
    raise ValueError(
        f"Note boundaries fall outside audio duration ({duration:.2f}s)."
    )

# Create visualization
fig, axes = plt.subplots(
    2,
    1,
    figsize=(16, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

# 1. Original audio waveform
axes[0].plot(
    time_axis,
    y,
    linewidth=0.5
)

axes[0].set_title("Original Humming Audio Waveform")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)

# 2. Detected note timeline
axes[1].set_title("Detected Note Boundaries")

for index, row in df.iterrows():

    start = row[start_col]
    end = row[end_col]
    note = str(row[note_col])

    axes[1].barh(
        y=0,
        width=end - start,
        left=start,
        height=0.5,
        edgecolor="black",
        alpha=0.75
    )

    axes[1].text(
        (start + end) / 2,
        0,
        note,
        ha="center",
        va="center",
        fontsize=8
    )

    # Mark note boundaries on waveform
    axes[0].axvline(
        start,
        linestyle="--",
        linewidth=0.6,
        alpha=0.4
    )

    axes[0].axvline(
        end,
        linestyle=":",
        linewidth=0.5,
        alpha=0.3
    )

axes[1].set_xlabel("Time (seconds)")
axes[1].set_yticks([])
axes[1].set_xlim(0, duration)
axes[1].grid(True, axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Summary
print("\nAUDIO AND NOTE ALIGNMENT ANALYSIS")
print("=" * 55)

print(f"Audio Duration: {duration:.2f} seconds")
print(f"Detected Notes: {len(df)}")
print(f"First Note Start: {df[start_col].iloc[0]:.3f} seconds")
print(f"Last Note End: {df[end_col].iloc[-1]:.3f} seconds")

print("\nVisualization saved to:")
print(OUTPUT_FILE)

print("\nAnalysis Completed.")