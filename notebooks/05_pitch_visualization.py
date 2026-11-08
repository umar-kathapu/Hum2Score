import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(r"D:\Musical-Score-Generation")

AUDIO_DIR = BASE_DIR / "processed_data" / "hum"

OUTPUT_DIR = BASE_DIR / "outputs" / "pitch_detection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Select one sample
audio_file = AUDIO_DIR / "0000.wav"

# Load audio
y, sr = librosa.load(audio_file, sr=22050)

# Pitch detection
f0, voiced_flag, voiced_prob = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr,
    frame_length=2048,
    hop_length=256
)

# Time axis
times = librosa.times_like(f0, sr=sr, hop_length=256)

# Convert frequencies to musical notes
valid_f0 = f0[voiced_flag]

notes = librosa.hz_to_note(valid_f0)

# Plot
plt.figure(figsize=(14, 6))

plt.scatter(
    times[voiced_flag],
    f0[voiced_flag],
    s=8,
    label="Detected Pitch"
)

plt.yscale("log")

plt.yticks(
    librosa.note_to_hz(["C2", "C3", "C4", "C5", "C6"]),
    ["C2", "C3", "C4", "C5", "C6"]
)

plt.xlabel("Time (Seconds)")
plt.ylabel("Musical Pitch")

plt.title("Pitch Contour - 0000.wav")

plt.grid(True)
plt.legend()

plt.savefig(
    OUTPUT_DIR / "pitch_contour_0000.png",
    dpi=300
)

plt.show()

print("Pitch visualization completed!")
print("Total detected frames:", len(valid_f0))
print("Unique notes:", np.unique(notes))