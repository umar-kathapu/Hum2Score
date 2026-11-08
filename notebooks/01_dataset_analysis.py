import os
import pandas as pd
import soundfile as sf
from pathlib import Path
from collections import Counter

# Dataset paths
BASE_DIR = Path(r"D:\Musical-Score-Generation\dataset")

AUDIO_CSV = BASE_DIR / "MLEndHWD_Audio_Attributes.csv"

# Load metadata
df = pd.read_csv(AUDIO_CSV)

print("=" * 50)
print("MUSICAL SCORE GENERATION - DATASET ANALYSIS")
print("=" * 50)

print("\n1. DATASET INFORMATION")
print("Total Records:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\n2. RECORDING TYPE DISTRIBUTION")
print(df["Interpretation"].value_counts())

print("\n3. SONG CATEGORY DISTRIBUTION")
print(df["Song"].value_counts())

# Find WAV files
wav_files = list(BASE_DIR.rglob("*.wav"))

print("\n4. AUDIO FILE INFORMATION")
print("Total WAV Files:", len(wav_files))

# Analyse audio properties
sample_rates = Counter()
channels = Counter()
durations = []

print("\nAnalysing audio files...")

for index, file in enumerate(wav_files):

    try:
        info = sf.info(file)

        sample_rates[info.samplerate] += 1
        channels[info.channels] += 1
        durations.append(info.duration)

    except Exception as e:
        print("Error:", file.name, e)

    if (index + 1) % 500 == 0:
        print(f"Processed {index + 1}/{len(wav_files)}")

print("\n5. AUDIO STATISTICS")
print("Sample Rates:", sample_rates)
print("Channels:", channels)

if durations:
    print("Average Duration:", round(sum(durations)/len(durations), 2))
    print("Minimum Duration:", round(min(durations), 2))
    print("Maximum Duration:", round(max(durations), 2))

print("\nDATASET ANALYSIS COMPLETED!")
