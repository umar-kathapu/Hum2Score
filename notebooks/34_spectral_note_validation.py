import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import find_peaks


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_FILE = PROJECT_ROOT / "processed_data" / "hum" / "0000.wav"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "spectral_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "spectral_validation_0000.csv"
PNG_OUTPUT = OUTPUT_DIR / "spectral_validation_0000.png"


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

SR = 22050

TARGET_NOTES = [
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

# Expected frequencies
EXPECTED_FREQUENCIES = {
    "F#3": 185.00,
    "D#3": 155.56
}


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
# 4. SPECTRAL ANALYSIS FUNCTION
# --------------------------------------------------

def analyze_note(target):

    label = target["label"]
    start = target["start"]
    end = target["end"]

    print(f"\nAnalyzing {label}...")

    # Extract note segment
    start_sample = int(start * sr)
    end_sample = int(end * sr)

    segment = y[start_sample:end_sample]

    if len(segment) < 256:
        print("Segment too short.")
        return None

    # Remove DC offset
    segment = segment - np.mean(segment)

    # Apply Hann window
    window = np.hanning(len(segment))

    windowed = segment * window

    # FFT with zero-padding for smoother spectrum display
    n_fft = 65536

    spectrum = np.abs(
        np.fft.rfft(
            windowed,
            n=n_fft
        )
    )

    frequencies = np.fft.rfftfreq(
        n_fft,
        d=1 / sr
    )

    # Restrict analysis to 70–1200 Hz
    valid = (
        (frequencies >= 70) &
        (frequencies <= 1200)
    )

    freq_range = frequencies[valid]
    magnitude_range = spectrum[valid]

    # Find prominent peaks
    peaks, properties = find_peaks(
        magnitude_range,
        prominence=np.max(magnitude_range) * 0.015,
        distance=20
    )

    sorted_peaks = peaks[
        np.argsort(magnitude_range[peaks])[::-1]
    ]

    top_peaks = sorted_peaks[:10]

    dominant_peaks = []

    for peak in top_peaks:

        dominant_peaks.append({
            "Frequency": freq_range[peak],
            "Magnitude": magnitude_range[peak]
        })

    # RMS energy
    rms = np.sqrt(np.mean(segment ** 2))

    # Expected note frequency
    expected_frequency = EXPECTED_FREQUENCIES[label]

    # Fundamental and harmonic magnitudes
    harmonic_results = []

    for harmonic in range(1, 7):

        target_frequency = expected_frequency * harmonic

        band = (
            (freq_range >= target_frequency - 8) &
            (freq_range <= target_frequency + 8)
        )

        if np.any(band):

            harmonic_magnitude = np.max(
                magnitude_range[band]
            )

        else:

            harmonic_magnitude = 0

        harmonic_results.append({
            "Harmonic": harmonic,
            "Expected Frequency": target_frequency,
            "Magnitude": harmonic_magnitude
        })

    fundamental_magnitude = harmonic_results[0]["Magnitude"]

    total_harmonic_magnitude = sum(
        item["Magnitude"]
        for item in harmonic_results
    )

    fundamental_ratio = (
        fundamental_magnitude / total_harmonic_magnitude
        if total_harmonic_magnitude > 0
        else 0
    )

    # Return results
    return {
        "label": label,
        "start": start,
        "end": end,
        "duration": end - start,
        "expected_frequency": expected_frequency,
        "rms": rms,
        "fundamental_ratio": fundamental_ratio,
        "frequencies": freq_range,
        "magnitudes": magnitude_range,
        "dominant_peaks": dominant_peaks,
        "harmonics": harmonic_results
    }


# --------------------------------------------------
# 5. ANALYZE BOTH NOTES
# --------------------------------------------------

results = []

for target in TARGET_NOTES:

    result = analyze_note(target)

    if result is not None:
        results.append(result)


# --------------------------------------------------
# 6. VISUALIZATION
# --------------------------------------------------

fig, axes = plt.subplots(
    len(results),
    2,
    figsize=(15, 9)
)

if len(results) == 1:
    axes = np.array([axes])

for index, result in enumerate(results):

    # Spectrum
    ax = axes[index, 0]

    ax.plot(
        result["frequencies"],
        result["magnitudes"]
    )

    ax.set_xlim(70, 1200)

    ax.axvline(
        result["expected_frequency"],
        linestyle="--",
        label="Expected Fundamental"
    )

    ax.set_title(
        f'{result["label"]} - Frequency Spectrum'
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Harmonic magnitudes
    ax = axes[index, 1]

    harmonic_numbers = [
        item["Harmonic"]
        for item in result["harmonics"]
    ]

    harmonic_magnitudes = [
        item["Magnitude"]
        for item in result["harmonics"]
    ]

    ax.bar(
        harmonic_numbers,
        harmonic_magnitudes
    )

    ax.set_title(
        f'{result["label"]} - Harmonic Analysis'
    )

    ax.set_xlabel("Harmonic Number")
    ax.set_ylabel("Magnitude")

    ax.grid(True, alpha=0.3)


plt.tight_layout()

plt.savefig(
    PNG_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 7. SAVE CSV RESULTS
# --------------------------------------------------

rows = []

for result in results:

    for peak in result["dominant_peaks"]:

        rows.append({
            "Note": result["label"],
            "Start Time": result["start"],
            "End Time": result["end"],
            "Duration": result["duration"],
            "Expected Frequency": result["expected_frequency"],
            "RMS Energy": result["rms"],
            "Fundamental Ratio": result["fundamental_ratio"],
            "Dominant Peak Frequency": peak["Frequency"],
            "Peak Magnitude": peak["Magnitude"]
        })

df = pd.DataFrame(rows)

df.to_csv(
    CSV_OUTPUT,
    index=False
)


# --------------------------------------------------
# 8. PRINT RESULTS
# --------------------------------------------------

print("\nSPECTRAL VALIDATION RESULTS")
print("=" * 65)

for result in results:

    print(f"\nNote: {result['label']}")

    print(
        f"Expected Frequency: "
        f"{result['expected_frequency']:.2f} Hz"
    )

    print(
        f"RMS Energy: {result['rms']:.6f}"
    )

    print(
        f"Fundamental Ratio: "
        f"{result['fundamental_ratio']:.4f}"
    )

    print("Dominant Spectral Peaks:")

    for peak in result["dominant_peaks"][:5]:

        print(
            f"  {peak['Frequency']:.2f} Hz "
            f"(Magnitude: {peak['Magnitude']:.4f})"
        )

print("\nFILES GENERATED")
print(CSV_OUTPUT)
print(PNG_OUTPUT)

print("\nSpectral Validation Completed.")