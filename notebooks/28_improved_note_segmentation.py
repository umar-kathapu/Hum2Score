import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.ndimage import median_filter


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_FILE = PROJECT_ROOT / "processed_data" / "hum" / "0000.wav"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "improved_segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "improved_notes_0000.csv"

PLOT_OUTPUT = OUTPUT_DIR / "improved_note_segmentation_0000.png"


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

SR = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 256

MEDIAN_FILTER_SIZE = 5

MIN_NOTE_DURATION = 0.08

# A small gap between voiced regions may be bridged.
MAX_BRIDGE_GAP = 0.04

# --------------------------------------------------
# 3. Load audio
# --------------------------------------------------

y, sr = librosa.load(
    AUDIO_FILE,
    sr=SR,
    mono=True
)

duration = len(y) / sr

print("\nIMPROVED NOTE SEGMENTATION")
print("=" * 60)

print(f"Audio Duration: {duration:.2f} seconds")


# --------------------------------------------------
# 4. Estimate pitch using pYIN
# --------------------------------------------------

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

# Use only voiced frames
valid_voiced = (
    voiced_flag &
    np.isfinite(f0) &
    (f0 > 0)
)

# Convert F0 to MIDI
midi_values = np.full(len(f0), np.nan)

midi_values[valid_voiced] = librosa.hz_to_midi(
    f0[valid_voiced]
)


# --------------------------------------------------
# 5. Smooth pitch fluctuations
# --------------------------------------------------

smoothed_midi = midi_values.copy()

valid_indices = np.where(valid_voiced)[0]

# Median filter only inside contiguous voiced regions.
# This avoids filtering across unvoiced gaps.
if len(valid_indices) > 0:

    groups = np.split(
        valid_indices,
        np.where(np.diff(valid_indices) > 1)[0] + 1
    )

    for group in groups:

        if len(group) == 0:
            continue

        values = midi_values[group]

        smoothed_midi[group] = median_filter(
            values,
            size=MEDIAN_FILTER_SIZE,
            mode="nearest"
        )


# --------------------------------------------------
# 6. Bridge very short unvoiced gaps
# --------------------------------------------------

voiced_mask = valid_voiced.copy()

max_gap_frames = int(
    MAX_BRIDGE_GAP * sr / HOP_LENGTH
)

for index in range(1, len(voiced_mask) - 1):

    if not voiced_mask[index]:

        left = index - 1
        right = index + 1

        while left >= 0 and not voiced_mask[left]:
            left -= 1

        while right < len(voiced_mask) and not voiced_mask[right]:
            right += 1

        gap_length = right - left - 1

        if (
            left >= 0
            and right < len(voiced_mask)
            and gap_length <= max_gap_frames
        ):

            # Bridge only if both sides have similar pitch.
            if abs(
                smoothed_midi[left] - smoothed_midi[right]
            ) <= 0.5:

                voiced_mask[index:right] = True

                # Interpolate pitch values in the gap.
                smoothed_midi[index:right] = np.interp(
                    times[index:right],
                    [times[left], times[right]],
                    [
                        smoothed_midi[left],
                        smoothed_midi[right]
                    ]
                )


# --------------------------------------------------
# 7. Convert smoothed pitch into note labels
# --------------------------------------------------

rounded_midi = np.full(
    len(smoothed_midi),
    np.nan
)

rounded_midi[voiced_mask] = np.rint(
    smoothed_midi[voiced_mask]
)

# --------------------------------------------------
# 8. Detect note boundaries
# --------------------------------------------------

notes = []

active_start = None
active_pitch = None

for index in range(len(rounded_midi)):

    current_pitch = rounded_midi[index]

    if np.isnan(current_pitch):

        if active_start is not None:

            start_time = times[active_start]
            end_time = times[index]

            note_duration = end_time - start_time

            if note_duration >= MIN_NOTE_DURATION:

                notes.append({
                    "Start Time": start_time,
                    "End Time": end_time,
                    "Duration": note_duration,
                    "MIDI": int(active_pitch),
                    "Note": librosa.midi_to_note(
                        int(active_pitch)
                    )
                })

            active_start = None
            active_pitch = None

        continue

    if active_start is None:

        active_start = index
        active_pitch = current_pitch

    elif current_pitch != active_pitch:

        start_time = times[active_start]
        end_time = times[index]

        note_duration = end_time - start_time

        if note_duration >= MIN_NOTE_DURATION:

            notes.append({
                "Start Time": start_time,
                "End Time": end_time,
                "Duration": note_duration,
                "MIDI": int(active_pitch),
                "Note": librosa.midi_to_note(
                    int(active_pitch)
                )
            })

        active_start = index
        active_pitch = current_pitch


# Handle final active note
if active_start is not None:

    start_time = times[active_start]
    end_time = min(duration, times[-1] + HOP_LENGTH / sr)

    note_duration = end_time - start_time

    if note_duration >= MIN_NOTE_DURATION:

        notes.append({
            "Start Time": start_time,
            "End Time": end_time,
            "Duration": note_duration,
            "MIDI": int(active_pitch),
            "Note": librosa.midi_to_note(
                int(active_pitch)
            )
        })


# --------------------------------------------------
# 9. Create and save DataFrame
# --------------------------------------------------

notes_df = pd.DataFrame(notes)

if notes_df.empty:
    raise ValueError("No notes detected.")

notes_df.to_csv(
    CSV_OUTPUT,
    index=False
)


# --------------------------------------------------
# 10. Visualization
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(15, 9)
)

# Pitch contour
axes[0].plot(
    times,
    midi_values,
    alpha=0.35,
    label="Original pYIN MIDI"
)

axes[0].plot(
    times,
    smoothed_midi,
    linewidth=1,
    label="Smoothed MIDI"
)

axes[0].set_title("Pitch Smoothing")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("MIDI Pitch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Note segmentation
for _, row in notes_df.iterrows():

    axes[1].barh(
        0,
        row["Duration"],
        left=row["Start Time"],
        height=0.5,
        edgecolor="black"
    )

    axes[1].text(
        row["Start Time"] + row["Duration"] / 2,
        0,
        row["Note"],
        ha="center",
        va="center",
        fontsize=8
    )

axes[1].set_title("Improved Note Segmentation")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_yticks([])
axes[1].set_xlim(0, duration)
axes[1].grid(True, axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOT_OUTPUT,
    dpi=300
)

plt.show()


# --------------------------------------------------
# 11. Summary
# --------------------------------------------------

print("\nSEGMENTATION SUMMARY")
print("=" * 60)

print(f"Original Audio Duration: {duration:.2f} seconds")
print(f"Detected Notes: {len(notes_df)}")

print(
    f"Average Note Duration: "
    f"{notes_df['Duration'].mean():.4f} seconds"
)

print(
    f"Median Note Duration: "
    f"{notes_df['Duration'].median():.4f} seconds"
)

print("\nFirst 15 Detected Notes:")
print(
    notes_df.head(15).to_string(index=False)
)

print("\nFiles Saved:")
print(CSV_OUTPUT)
print(PLOT_OUTPUT)

print("\nImproved Segmentation Completed.")