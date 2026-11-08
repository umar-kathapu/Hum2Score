import librosa
import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(r"D:\Musical-Score-Generation")

AUDIO_DIR = BASE_DIR / "processed_data" / "hum"

OUTPUT_DIR = BASE_DIR / "outputs" / "pitch_detection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Select 5 sample files
audio_files = sorted(AUDIO_DIR.glob("*.wav"))[:5]

results = []

print("=" * 55)
print("PYIN PITCH DETECTION")
print("=" * 55)

for file in audio_files:

    print(f"\nProcessing: {file.name}")

    try:

        # Load audio
        y, sr = librosa.load(file, sr=22050)

        # Estimate pitch using pYIN
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=2048,
            hop_length=256
        )

        # Remove unvoiced frames
        valid_f0 = f0[voiced_flag]

        if len(valid_f0) > 0:

            mean_pitch = np.mean(valid_f0)
            median_pitch = np.median(valid_f0)

            notes = librosa.hz_to_note(valid_f0)

            unique_notes = pd.Series(notes).value_counts()

            print("Voiced Frames:", len(valid_f0))
            print("Mean Frequency:", round(mean_pitch, 2), "Hz")
            print("Median Frequency:", round(median_pitch, 2), "Hz")
            print("Detected Notes:", unique_notes.head(10).to_dict())

            results.append({
                "Filename": file.name,
                "Mean Frequency": mean_pitch,
                "Median Frequency": median_pitch,
                "Voiced Frames": len(valid_f0),
                "Total Frames": len(f0),
                "Voiced Ratio": len(valid_f0) / len(f0)
            })

        else:

            print("No voiced pitch detected.")

    except Exception as e:

        print("Error:", e)

# Save results
df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_DIR / "pyin_sample_results.csv",
    index=False
)

print("\n" + "=" * 55)
print("PITCH DETECTION COMPLETED")
print("=" * 55)

print(df)
print("\nResults saved successfully!")