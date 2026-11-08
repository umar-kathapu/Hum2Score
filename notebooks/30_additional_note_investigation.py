import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_FILE = (
    PROJECT_ROOT / "processed_data" / "hum" / "0000.wav"
)

NOTES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "improved_segmentation"
    / "improved_notes_0000.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "segmentation_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR / "additional_note_investigation_0000.png"
)


# --------------------------------------------------
# 2. Additional note candidates
# --------------------------------------------------

TARGET_NOTES = [
    {
        "label": "Additional Note 1 - F3",
        "start": 4.098322,
        "end": 4.179592,
    },
    {
        "label": "Additional Note 2 - F#3",
        "start": 5.398639,
        "end": 5.491519,
    },
    {
        "label": "Additional Note 3 - D#3",
        "start": 10.042630,
        "end": 10.205170,
    }
]


# --------------------------------------------------
# 3. Load audio
# --------------------------------------------------

SR = 22050

y, sr = librosa.load(
    AUDIO_FILE,
    sr=SR,
    mono=True
)

duration = len(y) / sr

# Pitch estimation
f0, voiced_flag, voiced_probability = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr,
    frame_length=2048,
    hop_length=256
)

times = librosa.times_like(
    f0,
    sr=sr,
    hop_length=256
)


# --------------------------------------------------
# 4. Load note data
# --------------------------------------------------

notes_df = pd.read_csv(NOTES_FILE)


# --------------------------------------------------
# 5. Prepare visualization
# --------------------------------------------------

fig, axes = plt.subplots(
    3,
    3,
    figsize=(17, 13)
)

results = []


# --------------------------------------------------
# 6. Investigate each additional note
# --------------------------------------------------

for index, target in enumerate(TARGET_NOTES):

    start = target["start"]
    end = target["end"]

    label = target["label"]

    # Context around note
    context_start = max(0, start - 0.20)
    context_end = min(duration, end + 0.20)

    start_sample = int(context_start * sr)
    end_sample = int(context_end * sr)

    segment = y[start_sample:end_sample]

    segment_times = (
        np.arange(len(segment)) / sr
        + context_start
    )

    # Select pitch frames
    mask = (
        (times >= start) &
        (times < end) &
        np.isfinite(f0) &
        voiced_flag
    )

    note_f0 = f0[mask]

    note_probability = voiced_probability[mask]

    median_f0 = (
        np.median(note_f0)
        if len(note_f0) > 0
        else np.nan
    )

    median_probability = (
        np.median(note_probability)
        if len(note_probability) > 0
        else np.nan
    )

    estimated_note = (
        librosa.hz_to_note(median_f0)
        if np.isfinite(median_f0)
        else "No valid pitch"
    )

    # Save numerical results
    results.append({
        "Note": label,
        "Start Time": start,
        "End Time": end,
        "Duration": end - start,
        "Median F0": median_f0,
        "Estimated Pitch": estimated_note,
        "Voiced Frames": len(note_f0),
        "Median Voiced Probability": median_probability
    })

    # --------------------------------------------------
    # Waveform
    # --------------------------------------------------

    ax_wave = axes[index, 0]

    ax_wave.plot(
        segment_times,
        segment,
        linewidth=0.7
    )

    ax_wave.axvspan(
        start,
        end,
        alpha=0.2
    )

    ax_wave.set_title(f"{label} - Waveform")
    ax_wave.set_xlabel("Time (seconds)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True, alpha=0.3)

    # --------------------------------------------------
    # Spectrogram
    # --------------------------------------------------

    ax_spec = axes[index, 1]

    D = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(
                segment,
                n_fft=2048,
                hop_length=128
            )
        ),
        ref=np.max
    )

    librosa.display.specshow(
        D,
        sr=sr,
        hop_length=128,
        x_axis="time",
        y_axis="linear",
        ax=ax_spec
    )

    ax_spec.set_ylim(0, 1200)

    ax_spec.set_title(f"{label} - Spectrogram")
    ax_spec.set_xlabel("Time")
    ax_spec.set_ylabel("Frequency (Hz)")

    # --------------------------------------------------
    # Pitch contour
    # --------------------------------------------------

    ax_pitch = axes[index, 2]

    local_mask = (
        (times >= context_start) &
        (times <= context_end)
    )

    local_f0 = f0[local_mask]
    local_times = times[local_mask]

    ax_pitch.plot(
        local_times,
        local_f0,
        marker=".",
        markersize=3
    )

    ax_pitch.axvspan(
        start,
        end,
        alpha=0.2
    )

    ax_pitch.set_title(
        f"{label} - pYIN Pitch Contour"
    )

    ax_pitch.set_xlabel("Time (seconds)")
    ax_pitch.set_ylabel("Frequency (Hz)")
    ax_pitch.grid(True, alpha=0.3)


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

results_df = pd.DataFrame(results)

CSV_OUTPUT = OUTPUT_DIR / "additional_note_investigation_0000.csv"

results_df.to_csv(
    CSV_OUTPUT,
    index=False
)


# --------------------------------------------------
# 8. Print summary
# --------------------------------------------------

print("\nADDITIONAL NOTE INVESTIGATION")
print("=" * 65)

print(
    results_df.to_string(index=False)
)

print("\nFiles Saved:")
print(CSV_OUTPUT)
print(OUTPUT_FILE)

print("\nInvestigation Completed.")