import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_FILE = PROJECT_ROOT / "processed_data" / "hum" / "0000.wav"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pitch_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "autocorrelation_note27.png"
CSV_OUTPUT = OUTPUT_DIR / "autocorrelation_note27.csv"

# Note 27 boundaries
START_TIME = 9.508571428571429
END_TIME = 9.63628117913832

SR = 22050

# Load audio
y, sr = librosa.load(
    AUDIO_FILE,
    sr=SR,
    mono=True
)

# Extract note with context
padding = 0.10

start_sample = max(
    0,
    int((START_TIME - padding) * sr)
)

end_sample = min(
    len(y),
    int((END_TIME + padding) * sr)
)

segment = y[start_sample:end_sample]

# Select central region to reduce boundary effects
central_start = int(0.02 * sr)
central_end = min(
    len(segment),
    int(0.16 * sr)
)

signal = segment[central_start:central_end]

if len(signal) < 1000:
    raise ValueError("Insufficient audio samples for autocorrelation.")

# Remove DC component
signal = signal - np.mean(signal)

# Apply window
window = np.hanning(len(signal))
signal = signal * window

# Normalize
if np.max(np.abs(signal)) == 0:
    raise ValueError("Audio segment has zero amplitude.")

signal = signal / np.max(np.abs(signal))

# Calculate autocorrelation
autocorrelation = librosa.autocorrelate(signal)

# Normalize autocorrelation
autocorrelation = autocorrelation / autocorrelation[0]

# Search for candidate periods corresponding to 100-500 Hz
min_frequency = 100
max_frequency = 500

min_lag = int(sr / max_frequency)
max_lag = int(sr / min_frequency)

search_region = autocorrelation[min_lag:max_lag + 1]

peaks, properties = find_peaks(search_region)

if len(peaks) == 0:
    raise ValueError("No autocorrelation peaks detected.")

candidate_lags = peaks + min_lag

# Rank peaks by autocorrelation strength
ranked_indices = np.argsort(
    autocorrelation[candidate_lags]
)[::-1]

results = []

for rank, index in enumerate(ranked_indices[:10], start=1):

    lag = candidate_lags[index]

    frequency = sr / lag

    results.append({
        "Rank": rank,
        "Lag (Samples)": lag,
        "Frequency (Hz)": frequency,
        "Autocorrelation Strength": autocorrelation[lag]
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    CSV_OUTPUT,
    index=False
)

print("\nAUTOCORRELATION PITCH VALIDATION")
print("=" * 60)

print(f"Sample Rate: {sr} Hz")
print(f"Analysis Duration: {len(signal) / sr:.4f} seconds")

print("\nTop Candidate Frequencies:")
print(results_df.to_string(index=False))

# Reference estimates
print("\nReference Estimates:")
print("pYIN: 161.98 Hz")
print("YIN: 321.94 Hz")

# Plot autocorrelation
lags = np.arange(len(autocorrelation))
lag_times = lags / sr

plt.figure(figsize=(13, 6))

plt.plot(
    lag_times,
    autocorrelation
)

plt.xlim(
    0,
    1 / min_frequency
)

plt.xlabel("Lag (seconds)")
plt.ylabel("Normalized Autocorrelation")
plt.title("Autocorrelation Analysis - Note 27")

plt.grid(True, alpha=0.3)

for _, row in results_df.head(5).iterrows():

    lag_time = row["Lag (Samples)"] / sr

    plt.axvline(
        lag_time,
        linestyle="--",
        alpha=0.6,
        label=f"{row['Frequency (Hz)']:.1f} Hz"
    )

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300
)

plt.show()

print("\nFiles Saved:")
print(CSV_OUTPUT)
print(OUTPUT_FILE)

print("\nAnalysis Completed.")