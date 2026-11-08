import librosa
import soundfile as sf
import numpy as np

from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_FILE = PROJECT_ROOT / "processed_data" / "hum" / "0000.wav"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audio_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD AUDIO
# --------------------------------------------------

print("Loading original processed audio...")

y, sr = librosa.load(
    AUDIO_FILE,
    sr=None,
    mono=True
)

print(f"Sample Rate: {sr} Hz")
print(f"Duration: {len(y) / sr:.2f} seconds")


# --------------------------------------------------
# 3. DEFINE AUDIO REGIONS
# --------------------------------------------------

REGIONS = [
    {
        "name": "F_SHARP_3_REVIEW",
        "start": 5.10,
        "end": 5.70
    },
    {
        "name": "D_SHARP_3_REVIEW",
        "start": 9.80,
        "end": 10.50
    }
]


# --------------------------------------------------
# 4. EXTRACT AUDIO SEGMENTS
# --------------------------------------------------

print("\nEXTRACTING AUDIO SEGMENTS")
print("=" * 50)

for region in REGIONS:

    name = region["name"]
    start = region["start"]
    end = region["end"]

    start_sample = int(start * sr)
    end_sample = int(end * sr)

    segment = y[start_sample:end_sample]

    output_file = OUTPUT_DIR / f"{name}.wav"

    sf.write(
        output_file,
        segment,
        sr,
        subtype="PCM_16"
    )

    print(f"\nRegion: {name}")
    print(f"Start: {start:.2f} seconds")
    print(f"End: {end:.2f} seconds")
    print(f"Duration: {len(segment) / sr:.3f} seconds")
    print(f"Saved: {output_file}")


# --------------------------------------------------
# 5. COMPLETION
# --------------------------------------------------

print("\nAUDIO EXTRACTION COMPLETED!")

print("\nGenerated Files:")

for file in OUTPUT_DIR.glob("*.wav"):
    print(file)