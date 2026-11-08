import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_FILE = PROJECT_ROOT / "processed_data" / "hum" / "0000.wav"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "candidate_note_confidence_0000.csv"

PLOT_OUTPUT = OUTPUT_DIR / "candidate_note_confidence_0000.png"


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

SR = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 256

CANDIDATES = [
    {
        "label": "F3",
        "start": 4.098322,
        "end": 4.179592
    },
    {
        "label": "F#3",
        "start": 5.398639,
        "end": 5.491519
    },
    {
        "label": "D#3",
        "start": 10.042630,
        "end": 10.205170
    }
]


# --------------------------------------------------
# 3. Load audio
# --------------------------------------------------

y, sr = librosa.load(
    AUDIO_FILE,
    sr=SR,
    mono=True
)

print("\nCANDIDATE NOTE CONFIDENCE ANALYSIS")
print("=" * 65)

print(f"Audio Duration: {len(y) / sr:.2f} seconds")


# --------------------------------------------------
# 4. Estimate pYIN pitch
# --------------------------------------------------

pyin_f0, voiced_flag, voiced_probability = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
)

frame_times = librosa.times_like(
    pyin_f0,
    sr=sr,
    hop_length=HOP_LENGTH
)


# --------------------------------------------------
# 5. Estimate YIN pitch
# --------------------------------------------------

yin_f0 = librosa.yin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
)


# --------------------------------------------------
# 6. Calculate RMS energy
# --------------------------------------------------

rms = librosa.feature.rms(
    y=y,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
)[0]

# All feature arrays should have matching lengths.
min_length = min(
    len(frame_times),
    len(pyin_f0),
    len(yin_f0),
    len(voiced_probability),
    len(rms)
)

frame_times = frame_times[:min_length]
pyin_f0 = pyin_f0[:min_length]
yin_f0 = yin_f0[:min_length]
voiced_flag = voiced_flag[:min_length]
voiced_probability = voiced_probability[:min_length]
rms = rms[:min_length]


# --------------------------------------------------
# 7. Analyze individual candidates
# --------------------------------------------------

all_results = []
summary_results = []

for candidate in CANDIDATES:

    label = candidate["label"]
    start = candidate["start"]
    end = candidate["end"]

    mask = (
        (frame_times >= start) &
        (frame_times < end)
    )

    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(f"\nNo frames found for {label}")
        continue

    candidate_pyin = pyin_f0[indices]
    candidate_yin = yin_f0[indices]
    candidate_probability = voiced_probability[indices]
    candidate_rms = rms[indices]
    candidate_voiced = voiced_flag[indices]

    # Valid pYIN pitch estimates
    valid_pyin = (
        candidate_voiced &
        np.isfinite(candidate_pyin) &
        (candidate_pyin > 0)
    )

    valid_yin = (
        np.isfinite(candidate_yin) &
        (candidate_yin > 0)
    )

    # --------------------------------------------------
    # Frame-level results
    # --------------------------------------------------

    for local_index, global_index in enumerate(indices):

        all_results.append({
            "Candidate": label,
            "Frame Time": frame_times[global_index],
            "pYIN F0": pyin_f0[global_index],
            "YIN F0": yin_f0[global_index],
            "Voiced": bool(voiced_flag[global_index]),
            "Voiced Probability": voiced_probability[global_index],
            "RMS Energy": rms[global_index]
        })

    # --------------------------------------------------
    # Summary statistics
    # --------------------------------------------------

    valid_pyin_values = candidate_pyin[valid_pyin]
    valid_yin_values = candidate_yin[valid_yin]

    median_pyin = (
        np.median(valid_pyin_values)
        if len(valid_pyin_values) else np.nan
    )

    median_yin = (
        np.median(valid_yin_values)
        if len(valid_yin_values) else np.nan
    )

    median_probability = np.nanmedian(
        candidate_probability
    )

    median_rms = np.median(candidate_rms)

    # Count confidence levels
    high_confidence = np.sum(
        candidate_probability >= 0.8
    )

    medium_confidence = np.sum(
        (candidate_probability >= 0.5) &
        (candidate_probability < 0.8)
    )

    low_confidence = np.sum(
        candidate_probability < 0.5
    )

    summary_results.append({
        "Candidate": label,
        "Duration": end - start,
        "Total Frames": len(indices),
        "Voiced Frames": int(valid_pyin.sum()),
        "Voiced Percentage": (
            valid_pyin.sum() / len(indices) * 100
        ),
        "Median pYIN F0": median_pyin,
        "Median YIN F0": median_yin,
        "Median Voiced Probability": median_probability,
        "Median RMS Energy": median_rms,
        "High Confidence Frames": int(high_confidence),
        "Medium Confidence Frames": int(medium_confidence),
        "Low Confidence Frames": int(low_confidence)
    })


# --------------------------------------------------
# 8. Save CSV files
# --------------------------------------------------

frame_df = pd.DataFrame(all_results)
summary_df = pd.DataFrame(summary_results)

frame_df.to_csv(
    CSV_OUTPUT,
    index=False
)

SUMMARY_CSV = OUTPUT_DIR / "candidate_note_summary_0000.csv"

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# --------------------------------------------------
# 9. Print summary
# --------------------------------------------------

print("\nCANDIDATE SUMMARY")
print("-" * 65)

print(
    summary_df.to_string(index=False)
)

# --------------------------------------------------
# 10. Visualization
# --------------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(15, 11)
)

for index, candidate in enumerate(CANDIDATES):

    label = candidate["label"]
    start = candidate["start"]
    end = candidate["end"]

    local = frame_df[
        (frame_df["Candidate"] == label)
    ]

    if local.empty:
        continue

    ax = axes[index]

    # pYIN pitch
    ax.plot(
        local["Frame Time"],
        local["pYIN F0"],
        marker="o",
        label="pYIN F0"
    )

    # YIN pitch
    ax.plot(
        local["Frame Time"],
        local["YIN F0"],
        marker="x",
        label="YIN F0"
    )

    ax.set_title(
        f"{label} - Pitch and Confidence"
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")

    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()

    ax2.plot(
        local["Frame Time"],
        local["Voiced Probability"],
        linestyle=":",
        label="Voiced Probability"
    )

    ax2.set_ylabel("Voiced Probability")
    ax2.set_ylim(0, 1.05)

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

plt.tight_layout()

plt.savefig(
    PLOT_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 11. Final output
# --------------------------------------------------

print("\nFILES SAVED")
print("=" * 50)

print(CSV_OUTPUT)
print(SUMMARY_CSV)
print(PLOT_OUTPUT)

print("\nCandidate Confidence Analysis Completed.")