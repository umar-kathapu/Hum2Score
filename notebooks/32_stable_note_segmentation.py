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

OLD_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "note_segmentation"
    / "smoothed_notes_0000.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stable_segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "stable_notes_0000.csv"

PLOT_OUTPUT = OUTPUT_DIR / "stable_note_segmentation_0000.png"


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

SR = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 256

MEDIAN_FILTER_SIZE = 5

MIN_NOTE_DURATION = 0.08

# Remove extremely low-energy frames.
RMS_PERCENTILE = 15

# Remove very short pitch-label fluctuations.
MIN_LABEL_DURATION = 0.035

# Small gaps between same-pitch regions may be bridged.
MAX_SAME_NOTE_GAP = 0.035


# --------------------------------------------------
# 3. Load audio
# --------------------------------------------------

y, sr = librosa.load(
    AUDIO_FILE,
    sr=SR,
    mono=True
)

duration = len(y) / sr

print("\nSTABLE NOTE SEGMENTATION")
print("=" * 65)

print(f"Audio Duration: {duration:.2f} seconds")


# --------------------------------------------------
# 4. Pitch estimation using pYIN
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


# --------------------------------------------------
# 5. Calculate RMS energy
# --------------------------------------------------

rms = librosa.feature.rms(
    y=y,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
)[0]

# Align feature lengths safely.
frame_count = min(
    len(f0),
    len(voiced_flag),
    len(voiced_probability),
    len(rms),
    len(times)
)

f0 = f0[:frame_count]
voiced_flag = voiced_flag[:frame_count]
voiced_probability = voiced_probability[:frame_count]
rms = rms[:frame_count]
times = times[:frame_count]


# --------------------------------------------------
# 6. Create valid pitch mask
# --------------------------------------------------

positive_rms = rms[rms > 0]

if len(positive_rms) == 0:
    raise ValueError("No usable audio energy detected.")

energy_threshold = np.percentile(
    positive_rms,
    RMS_PERCENTILE
)

valid_mask = (
    voiced_flag
    & np.isfinite(f0)
    & (f0 > 0)
    & (rms >= energy_threshold)
)

print(f"RMS Energy Threshold: {energy_threshold:.6f}")

# Convert valid F0 to MIDI.
midi_values = np.full(frame_count, np.nan)

midi_values[valid_mask] = librosa.hz_to_midi(
    f0[valid_mask]
)


# --------------------------------------------------
# 7. Smooth pitch inside voiced regions
# --------------------------------------------------

smoothed_midi = midi_values.copy()

valid_indices = np.where(valid_mask)[0]

groups = np.split(
    valid_indices,
    np.where(np.diff(valid_indices) > 1)[0] + 1
)

for group in groups:

    if len(group) == 0:
        continue

    smoothed_midi[group] = median_filter(
        midi_values[group],
        size=MEDIAN_FILTER_SIZE,
        mode="nearest"
    )


# --------------------------------------------------
# 8. Convert pitch to semitone labels
# --------------------------------------------------

rounded_midi = np.full(frame_count, np.nan)

rounded_midi[valid_mask] = np.rint(
    smoothed_midi[valid_mask]
)


# --------------------------------------------------
# 9. Remove very short pitch-label fluctuations
# --------------------------------------------------

minimum_frames = max(
    1,
    int(np.ceil(
        MIN_LABEL_DURATION * sr / HOP_LENGTH
    ))
)

labels = rounded_midi.copy()

# Find contiguous valid regions.
valid_indices = np.where(valid_mask)[0]

groups = np.split(
    valid_indices,
    np.where(np.diff(valid_indices) > 1)[0] + 1
)

for group in groups:

    if len(group) == 0:
        continue

    local_labels = labels[group].copy()

    changed = True

    while changed:

        changed = False

        boundaries = np.r_[
            0,
            np.where(
                np.diff(local_labels) != 0
            )[0] + 1,
            len(local_labels)
        ]

        runs = [
            (
                boundaries[i],
                boundaries[i + 1],
                local_labels[boundaries[i]]
            )
            for i in range(len(boundaries) - 1)
        ]

        for run_index, (start, end, pitch) in enumerate(runs):

            run_length = end - start

            if run_length >= minimum_frames:
                continue

            # Replace a short internal pitch excursion
            # only when both neighbouring pitches agree.
            if (
                run_index > 0
                and run_index < len(runs) - 1
            ):

                previous_pitch = runs[run_index - 1][2]
                next_pitch = runs[run_index + 1][2]

                if previous_pitch == next_pitch:

                    local_labels[start:end] = previous_pitch
                    changed = True
                    break

    labels[group] = local_labels


# --------------------------------------------------
# 10. Build note segments
# --------------------------------------------------

notes = []

active_start = None
active_pitch = None

for index in range(frame_count):

    current_pitch = labels[index]

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


# Handle final active note.
if active_start is not None:

    start_time = times[active_start]

    end_time = min(
        duration,
        times[-1] + HOP_LENGTH / sr
    )

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
# 11. Merge very small same-pitch gaps
# --------------------------------------------------

merged_notes = []

for note in notes:

    if not merged_notes:

        merged_notes.append(note.copy())
        continue

    previous = merged_notes[-1]

    gap = note["Start Time"] - previous["End Time"]

    if (
        note["MIDI"] == previous["MIDI"]
        and 0 <= gap <= MAX_SAME_NOTE_GAP
    ):

        previous["End Time"] = note["End Time"]

        previous["Duration"] = (
            previous["End Time"] - previous["Start Time"]
        )

    else:

        merged_notes.append(note.copy())


# --------------------------------------------------
# 12. Save results
# --------------------------------------------------

notes_df = pd.DataFrame(merged_notes)

if notes_df.empty:
    raise ValueError("No stable notes detected.")

notes_df.to_csv(
    CSV_OUTPUT,
    index=False
)


# --------------------------------------------------
# 13. Visualization
# --------------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(15, 11),
    sharex=True
)

# Original pitch
axes[0].plot(
    times,
    midi_values,
    linewidth=0.8,
    label="Original MIDI"
)

axes[0].set_title("Original pYIN Pitch")
axes[0].set_ylabel("MIDI Pitch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)


# Smoothed pitch
axes[1].plot(
    times,
    smoothed_midi,
    linewidth=1,
    label="Smoothed MIDI"
)

axes[1].plot(
    times,
    labels,
    linewidth=1,
    label="Stable Note Labels"
)

axes[1].set_title("Stable Pitch Segmentation")
axes[1].set_ylabel("MIDI Pitch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)


# Final note segments
for _, row in notes_df.iterrows():

    axes[2].barh(
        0,
        row["Duration"],
        left=row["Start Time"],
        height=0.5,
        edgecolor="black"
    )

    axes[2].text(
        row["Start Time"] + row["Duration"] / 2,
        0,
        row["Note"],
        ha="center",
        va="center",
        fontsize=8
    )

axes[2].set_title(
    f"Final Stable Notes ({len(notes_df)} Notes)"
)

axes[2].set_ylabel("Notes")
axes[2].set_xlabel("Time (seconds)")
axes[2].set_yticks([])
axes[2].set_xlim(0, duration)
axes[2].grid(True, axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOT_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 14. Summary
# --------------------------------------------------

print("\nSEGMENTATION SUMMARY")
print("=" * 65)

print(f"Detected Stable Notes: {len(notes_df)}")

print(
    f"Average Duration: "
    f"{notes_df['Duration'].mean():.4f} seconds"
)

print(
    f"Median Duration: "
    f"{notes_df['Duration'].median():.4f} seconds"
)

print("\nFirst 20 Notes:")

print(
    notes_df.head(20).to_string(index=False)
)


# --------------------------------------------------
# 15. Compare with previous versions
# --------------------------------------------------

print("\nCOMPARISON WITH PREVIOUS VERSIONS")
print("-" * 65)

if OLD_FILE.exists():

    old_df = pd.read_csv(OLD_FILE)

    print(
        f"Previous Segmentation: {len(old_df)} notes"
    )

print(
    f"Improved Segmentation V1: 50 notes"
)

print(
    f"Stable Segmentation V2: {len(notes_df)} notes"
)


print("\nFILES SAVED")
print("=" * 50)

print(CSV_OUTPUT)
print(PLOT_OUTPUT)

print("\nStable Note Segmentation Completed.")