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

CSV_OUTPUT = OUTPUT_DIR / "independent_pitch_validation_0000.csv"
PLOT_OUTPUT = OUTPUT_DIR / "independent_pitch_validation_0000.png"

# Configuration
SR = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 256

# Load audio
y, sr = librosa.load(
    AUDIO_FILE,
    sr=SR,
    mono=True
)

# Load existing detected notes
df = pd.read_csv(NOTES_FILE)

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

if not all([start_col, end_col, note_col]):
    raise ValueError(
        f"Required columns missing: {list(df.columns)}"
    )

df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
df[end_col] = pd.to_numeric(df[end_col], errors="coerce")

df = df.dropna(
    subset=[start_col, end_col, note_col]
).sort_values(start_col).reset_index(drop=True)

# --------------------------------------------------
# 1. Estimate pitch using YIN
# --------------------------------------------------

yin_f0 = librosa.yin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
)

yin_times = librosa.times_like(
    yin_f0,
    sr=sr,
    hop_length=HOP_LENGTH
)

# --------------------------------------------------
# 2. Estimate pitch using pYIN
# --------------------------------------------------

pyin_f0, pyin_voiced, pyin_probability = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
)

pyin_times = librosa.times_like(
    pyin_f0,
    sr=sr,
    hop_length=HOP_LENGTH
)

# --------------------------------------------------
# 3. Compare YIN and pYIN within each note segment
# --------------------------------------------------

def frequency_to_midi(frequency):
    return librosa.hz_to_midi(frequency)

results = []

for index, row in df.iterrows():

    start = float(row[start_col])
    end = float(row[end_col])
    detected_note = str(row[note_col])

    detected_note = detected_note.replace("♯", "#")

    detected_midi = librosa.note_to_midi(detected_note)

    yin_mask = (
        (yin_times >= start) &
        (yin_times < end) &
        np.isfinite(yin_f0)
    )

    pyin_mask = (
        (pyin_times >= start) &
        (pyin_times < end) &
        np.isfinite(pyin_f0)
    )

    segment_yin = yin_f0[yin_mask]
    segment_pyin = pyin_f0[pyin_mask]

    median_yin = (
        np.median(segment_yin)
        if len(segment_yin) > 0 else np.nan
    )

    median_pyin = (
        np.median(segment_pyin)
        if len(segment_pyin) > 0 else np.nan
    )

    yin_midi = (
        frequency_to_midi(median_yin)
        if np.isfinite(median_yin) else np.nan
    )

    pyin_midi = (
        frequency_to_midi(median_pyin)
        if np.isfinite(median_pyin) else np.nan
    )

    results.append({
        "Note Number": index + 1,
        "Detected Note": detected_note,
        "Start Time": start,
        "End Time": end,
        "YIN F0": median_yin,
        "pYIN F0": median_pyin,
        "YIN MIDI": yin_midi,
        "pYIN MIDI": pyin_midi,
        "YIN Error (Cents)": (
            (yin_midi - detected_midi) * 100
            if np.isfinite(yin_midi) else np.nan
        ),
        "pYIN Error (Cents)": (
            (pyin_midi - detected_midi) * 100
            if np.isfinite(pyin_midi) else np.nan
        ),
        "YIN-pYIN Difference (Cents)": (
            (yin_midi - pyin_midi) * 100
            if np.isfinite(yin_midi) and np.isfinite(pyin_midi)
            else np.nan
        ),
        "YIN Frames": len(segment_yin),
        "pYIN Frames": len(segment_pyin)
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    CSV_OUTPUT,
    index=False
)

# --------------------------------------------------
# 4. Print comparison statistics
# --------------------------------------------------

valid = results_df.dropna(
    subset=["YIN-pYIN Difference (Cents)"]
)

print("\nINDEPENDENT PITCH VALIDATION")
print("=" * 65)

print(f"Total Detected Notes: {len(results_df)}")
print(f"Notes with both estimates: {len(valid)}")

if len(valid) > 0:

    differences = valid["YIN-pYIN Difference (Cents)"].abs()

    print("\nYIN vs pYIN Difference:")

    print(
        f"Mean Absolute Difference: "
        f"{differences.mean():.2f} cents"
    )

    print(
        f"Median Absolute Difference: "
        f"{differences.median():.2f} cents"
    )

    print(
        f"Maximum Difference: "
        f"{differences.max():.2f} cents"
    )

    for threshold in [25, 50, 100, 200]:
        count = (differences <= threshold).sum()

        print(
            f"Within {threshold} cents: "
            f"{count}/{len(valid)} "
            f"({count / len(valid) * 100:.2f}%)"
        )

    # Potential octave disagreements
    octave_disagreements = valid[
        differences.between(1100, 1300)
    ]

    print(
        f"\nPotential Octave Disagreements: "
        f"{len(octave_disagreements)}"
    )

    # --------------------------------------------------
    # 5. Visualization
    # --------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.plot(
        valid["Note Number"],
        valid["YIN-pYIN Difference (Cents)"],
        marker="o",
        label="YIN - pYIN"
    )

    plt.axhline(0, linestyle="--")

    plt.axhline(50, linestyle=":")
    plt.axhline(-50, linestyle=":")

    plt.title("Independent Pitch Comparison: YIN vs pYIN")
    plt.xlabel("Note Number")
    plt.ylabel("Pitch Difference (Cents)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOT_OUTPUT,
        dpi=300
    )

    plt.show()

else:
    print("\nInsufficient overlapping pitch estimates.")

print("\nFiles Saved:")
print(CSV_OUTPUT)

if len(valid) > 0:
    print(PLOT_OUTPUT)

print("\nAnalysis Completed.")