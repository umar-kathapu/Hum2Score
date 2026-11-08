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

CSV_OUTPUT = OUTPUT_DIR / "pitch_accuracy_0000.csv"
PLOT_OUTPUT = OUTPUT_DIR / "pitch_accuracy_0000.png"

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

# Load detected notes
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

# Estimate fundamental frequency using pYIN
f0, voiced_flag, voiced_probability = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
)

times = librosa.times_like(
    f0,
    sr=sr,
    hop_length=HOP_LENGTH
)

# Convert detected notes into MIDI numbers
def note_to_midi(note_name):
    note_name = str(note_name).replace("♯", "#")
    return librosa.note_to_midi(note_name)

results = []

for index, row in df.iterrows():

    start = float(row[start_col])
    end = float(row[end_col])
    note_name = str(row[note_col])

    detected_midi = note_to_midi(note_name)

    # Select pYIN frames inside the note segment
    mask = (
        (times >= start) &
        (times < end) &
        np.isfinite(f0)
    )

    segment_f0 = f0[mask]
    segment_probability = voiced_probability[mask]

    if len(segment_f0) == 0:

        results.append({
            "Note Number": index + 1,
            "Note": note_name,
            "Start Time": start,
            "End Time": end,
            "Duration": end - start,
            "Voiced Frames": 0,
            "Median F0": np.nan,
            "Estimated MIDI": np.nan,
            "Detected MIDI": detected_midi,
            "Pitch Error (Semitones)": np.nan,
            "Pitch Error (Cents)": np.nan,
            "Median Voiced Probability": np.nan
        })

        continue

    # Median F0 reduces the effect of outlier frames
    median_f0 = np.median(segment_f0)

    estimated_midi = librosa.hz_to_midi(median_f0)

    pitch_error = estimated_midi - detected_midi

    results.append({
        "Note Number": index + 1,
        "Note": note_name,
        "Start Time": start,
        "End Time": end,
        "Duration": end - start,
        "Voiced Frames": len(segment_f0),
        "Median F0": median_f0,
        "Estimated MIDI": estimated_midi,
        "Detected MIDI": detected_midi,
        "Pitch Error (Semitones)": pitch_error,
        "Pitch Error (Cents)": pitch_error * 100,
        "Median Voiced Probability": np.median(segment_probability)
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv(
    CSV_OUTPUT,
    index=False
)

# Statistics
valid = results_df.dropna(
    subset=["Pitch Error (Cents)"]
)

print("\nPITCH ACCURACY VALIDATION")
print("=" * 60)

print(f"Total Detected Notes: {len(results_df)}")
print(f"Notes With Valid F0: {len(valid)}")
print(
    f"Notes Without Valid F0: "
    f"{len(results_df) - len(valid)}"
)

if len(valid) > 0:

    errors = valid["Pitch Error (Cents)"].abs()

    print("\nPitch Deviation Statistics:")
    print(f"Mean Absolute Error: {errors.mean():.2f} cents")
    print(f"Median Absolute Error: {errors.median():.2f} cents")
    print(f"Maximum Absolute Error: {errors.max():.2f} cents")

    print("\nPitch Agreement (Exploratory):")

    for threshold in [25, 50, 100, 200]:
        count = (errors <= threshold).sum()
        percentage = count / len(valid) * 100

        print(
            f"Within {threshold} cents: "
            f"{count}/{len(valid)} ({percentage:.2f}%)"
        )

    # Potential octave errors
    octave_errors = valid[
        valid["Pitch Error (Cents)"].abs().between(1100, 1300)
    ]

    print(
        f"\nPotential Octave Errors: {len(octave_errors)}"
    )

    # Plot pitch deviations
    plt.figure(figsize=(14, 6))

    plt.bar(
        valid["Note Number"],
        valid["Pitch Error (Cents)"]
    )

    plt.axhline(0, linestyle="--")

    plt.axhline(50, linestyle=":")
    plt.axhline(-50, linestyle=":")

    plt.title("Pitch Deviation Between Detected Notes and pYIN F0")
    plt.xlabel("Note Number")
    plt.ylabel("Pitch Error (Cents)")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOT_OUTPUT,
        dpi=300
    )

    plt.show()

else:
    print("\nNo valid pitch measurements available.")

print("\nFiles Saved:")
print(CSV_OUTPUT)

if len(valid) > 0:
    print(PLOT_OUTPUT)

print("\nValidation Completed.")