import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(r"D:\Musical-Score-Generation")

AUDIO_FILE = BASE_DIR / "processed_data" / "hum" / "0000.wav"

OUTPUT_DIR = BASE_DIR / "outputs" / "rhythm_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load audio
y, sr = librosa.load(AUDIO_FILE, sr=22050)

# Detect onset strength
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# Estimate BPM and beats
tempo, beat_frames = librosa.beat.beat_track(
    onset_envelope=onset_env,
    sr=sr
)

beat_times = librosa.frames_to_time(
    beat_frames,
    sr=sr
)

# Create time axis
times = librosa.times_like(y, sr=sr)

# Plot waveform
plt.figure(figsize=(16, 6))

librosa.display.waveshow(
    y,
    sr=sr,
    alpha=0.6
)

# Mark detected beats
for beat_time in beat_times:
    plt.axvline(
        x=beat_time,
        linestyle="--",
        linewidth=1,
        alpha=0.7
    )

plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")

estimated_bpm = float(np.asarray(tempo).reshape(-1)[0])

plt.title(f"Beat Detection - Estimated BPM: {estimated_bpm:.2f}")

plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save graph
output_file = OUTPUT_DIR / "beat_visualization_0000.png"

plt.savefig(output_file, dpi=300)
plt.show()

print("\nBEAT VISUALIZATION COMPLETED!")
print("Estimated BPM:", round(estimated_bpm, 2))
print("Detected Beats:", len(beat_times))
print("Graph saved at:", output_file)