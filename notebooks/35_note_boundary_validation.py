import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_FILE = PROJECT_ROOT / "processed_data" / "hum" / "0000.wav"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "boundary_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "note_boundary_validation_0000.csv"
PNG_OUTPUT = OUTPUT_DIR / "note_boundary_validation_0000.png"


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

SR = 22050
HOP_LENGTH = 128
FRAME_LENGTH = 2048

TARGET_NOTES = [
    {
        "label": "F#3",
        "start": 5.398639,
        "end": 5.491519,
        "expected_frequency": 185.00
    },
    {
        "label": "D#3",
        "start": 10.042630,
        "end": 10.205170,
        "expected_frequency": 155.56
    }
]

CONTEXT = 0.30


# --------------------------------------------------
# 3. LOAD AUDIO
# --------------------------------------------------

print("Loading audio...")

y, sr = librosa.load(
    AUDIO_FILE,
    sr=SR,
    mono=True
)

print(f"Audio duration: {len(y) / sr:.2f} seconds")


# --------------------------------------------------
# 4. PITCH DETECTION
# --------------------------------------------------

print("Running pYIN pitch detection...")

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

# Convert detected frequencies to MIDI
midi_values = np.full(len(f0), np.nan)

valid = np.isfinite(f0) & voiced_flag

midi_values[valid] = librosa.hz_to_midi(f0[valid])


# --------------------------------------------------
# 5. FRAME-LEVEL DATA
# --------------------------------------------------

frame_data = pd.DataFrame({
    "Time": times,
    "Frequency": f0,
    "MIDI": midi_values,
    "Voiced": voiced_flag,
    "Voiced Probability": voiced_probability
})


# --------------------------------------------------
# 6. BOUNDARY ANALYSIS
# --------------------------------------------------

results = []

fig, axes = plt.subplots(
    len(TARGET_NOTES),
    2,
    figsize=(16, 10)
)

for index, target in enumerate(TARGET_NOTES):

    label = target["label"]
    start = target["start"]
    end = target["end"]

    expected = target["expected_frequency"]

    context_start = max(0, start - CONTEXT)
    context_end = min(len(y) / sr, end + CONTEXT)

    local = frame_data[
        (frame_data["Time"] >= context_start) &
        (frame_data["Time"] <= context_end)
    ].copy()

    # Divide frames into before, during and after
    before = local[
        local["Time"] < start
    ]

    during = local[
        (local["Time"] >= start) &
        (local["Time"] < end)
    ]

    after = local[
        local["Time"] >= end
    ]

    def get_median_frequency(data):

        values = data["Frequency"].dropna()

        if len(values) == 0:
            return np.nan

        return float(values.median())

    before_f0 = get_median_frequency(before)
    during_f0 = get_median_frequency(during)
    after_f0 = get_median_frequency(after)

    def get_note_name(freq):

        if not np.isfinite(freq):
            return "Unvoiced"

        return librosa.hz_to_note(freq)

    before_note = get_note_name(before_f0)
    during_note = get_note_name(during_f0)
    after_note = get_note_name(after_f0)

    # Calculate pitch stability
    during_midi = during["MIDI"].dropna()

    if len(during_midi) > 1:

        midi_std = float(during_midi.std())

    else:

        midi_std = np.nan

    # Voiced probability
    median_probability = (
        float(during["Voiced Probability"].median())
        if len(during) > 0
        else np.nan
    )

    # Boundary frequency differences
    before_difference = (
        1200 * np.log2(before_f0 / expected)
        if np.isfinite(before_f0) and before_f0 > 0
        else np.nan
    )

    after_difference = (
        1200 * np.log2(after_f0 / expected)
        if np.isfinite(after_f0) and after_f0 > 0
        else np.nan
    )

    results.append({

        "Note": label,
        "Start Time": start,
        "End Time": end,
        "Duration": end - start,

        "Before Median F0": before_f0,
        "Before Pitch": before_note,

        "During Median F0": during_f0,
        "During Pitch": during_note,

        "After Median F0": after_f0,
        "After Pitch": after_note,

        "During MIDI Std": midi_std,
        "Median Voiced Probability": median_probability,

        "Before Difference (cents)": before_difference,
        "After Difference (cents)": after_difference,

        "Before Frames": len(before),
        "During Frames": len(during),
        "After Frames": len(after)

    })

    # --------------------------------------------------
    # 7. PITCH CONTOUR
    # --------------------------------------------------

    ax = axes[index, 0]

    ax.plot(
        local["Time"],
        local["Frequency"],
        marker=".",
        markersize=3,
        linewidth=1
    )

    ax.axvspan(
        start,
        end,
        alpha=0.2,
        label="Candidate Note"
    )

    ax.axhline(
        expected,
        linestyle="--",
        label="Expected Frequency"
    )

    ax.axvline(start, linestyle=":")
    ax.axvline(end, linestyle=":")

    ax.set_title(f"{label} - Pitch Boundary Analysis")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")

    ax.legend()
    ax.grid(True, alpha=0.3)

    # --------------------------------------------------
    # 8. MIDI PITCH CONTOUR
    # --------------------------------------------------

    ax = axes[index, 1]

    ax.plot(
        local["Time"],
        local["MIDI"],
        marker=".",
        markersize=3,
        linewidth=1
    )

    expected_midi = librosa.hz_to_midi(expected)

    ax.axhline(
        expected_midi,
        linestyle="--",
        label="Expected MIDI"
    )

    ax.axvspan(
        start,
        end,
        alpha=0.2,
        label="Candidate Note"
    )

    ax.axvline(start, linestyle=":")
    ax.axvline(end, linestyle=":")

    ax.set_title(f"{label} - MIDI Transition")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MIDI Number")

    ax.legend()
    ax.grid(True, alpha=0.3)


# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

plt.tight_layout()

plt.savefig(
    PNG_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

results_df = pd.DataFrame(results)

results_df.to_csv(
    CSV_OUTPUT,
    index=False
)


# --------------------------------------------------
# 10. PRINT SUMMARY
# --------------------------------------------------

print("\nNOTE BOUNDARY VALIDATION")
print("=" * 75)

print(
    results_df.to_string(index=False)
)

print("\nFILES GENERATED")
print(CSV_OUTPUT)
print(PNG_OUTPUT)

print("\nBoundary Validation Completed.")