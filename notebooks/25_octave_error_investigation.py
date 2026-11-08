import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_FILE = PROJECT_ROOT / "processed_data" / "hum" / "0000.wav"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pitch_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "note_27_octave_investigation.png"

# Note 27 boundaries
START_TIME = 9.508571428571429
END_TIME = 9.63628117913832

# Load audio
y, sr = librosa.load(
    AUDIO_FILE,
    sr=22050,
    mono=True
)

# Extract note segment with a little context
padding = 0.15

start_sample = max(0, int((START_TIME - padding) * sr))
end_sample = min(len(y), int((END_TIME + padding) * sr))

segment = y[start_sample:end_sample]

segment_time = (
    np.arange(len(segment)) / sr
    + start_sample / sr
)

# Frequency analysis
n_fft = 8192

spectrum = np.abs(
    librosa.stft(
        segment,
        n_fft=n_fft,
        hop_length=128,
        window="hann"
    )
)

frequencies = librosa.fft_frequencies(
    sr=sr,
    n_fft=n_fft
)

# Average spectrum
mean_spectrum = np.mean(spectrum, axis=1)

# Focus on relevant frequency range
mask = (
    (frequencies >= 70) &
    (frequencies <= 1000)
)

# Find strongest spectral peaks
from scipy.signal import find_peaks

peaks, _ = find_peaks(mean_spectrum[mask])

masked_frequencies = frequencies[mask]
masked_spectrum = mean_spectrum[mask]

sorted_peaks = peaks[
    np.argsort(masked_spectrum[peaks])[::-1]
]

print("\nNOTE 27 OCTAVE INVESTIGATION")
print("=" * 55)

print("Strongest spectral peaks:")

for peak in sorted_peaks[:10]:
    print(
        f"Frequency: {masked_frequencies[peak]:.2f} Hz | "
        f"Magnitude: {masked_spectrum[peak]:.4f}"
    )

# Plot waveform and spectrum
fig, axes = plt.subplots(
    2,
    1,
    figsize=(13, 8)
)

axes[0].plot(segment_time, segment)

axes[0].set_title("Note 27 - Audio Waveform")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    masked_frequencies,
    masked_spectrum
)

axes[1].axvline(
    161.98,
    linestyle="--",
    label="pYIN: 161.98 Hz"
)

axes[1].axvline(
    321.94,
    linestyle="--",
    label="YIN: 321.94 Hz"
)

axes[1].set_title("Note 27 - Average Frequency Spectrum")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300
)

plt.show()

print("\nSaved:")
print(OUTPUT_FILE)