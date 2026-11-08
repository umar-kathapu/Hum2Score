import librosa
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(r"D:\Musical-Score-Generation")

AUDIO_FILE = (
    BASE_DIR / "processed_data" / "hum" / "0000.wav"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "rhythm_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load audio
y, sr = librosa.load(AUDIO_FILE, sr=22050)

# Calculate onset strength
onset_env = librosa.onset.onset_strength(
    y=y,
    sr=sr
)

# Estimate tempo and beat positions
tempo, beat_frames = librosa.beat.beat_track(
    onset_envelope=onset_env,
    sr=sr,
    units="frames"
)

# Convert beat frames into seconds
beat_times = librosa.frames_to_time(
    beat_frames,
    sr=sr
)

# Convert tempo into a normal Python number
estimated_bpm = float(np.asarray(tempo).reshape(-1)[0])

# Display results
print("\nBPM ESTIMATION COMPLETED!")
print("--------------------------------")

print(f"Estimated Tempo: {estimated_bpm:.2f} BPM")
print("Detected Beat Count:", len(beat_times))

print("\nDetected Beat Times:")
print(np.round(beat_times, 3))

# Save results
output_file = OUTPUT_DIR / "bpm_estimation_0000.txt"

with open(output_file, "w") as file:
    file.write(f"Estimated BPM: {estimated_bpm:.2f}\n")
    file.write(f"Detected Beat Count: {len(beat_times)}\n")
    file.write(f"Beat Times: {np.round(beat_times, 3)}\n")

print("\nResults saved at:")
print(output_file)