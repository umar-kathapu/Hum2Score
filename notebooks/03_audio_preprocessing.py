import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==================================
# CONFIGURATION
# ==================================

BASE_DIR = Path(r"D:\Musical-Score-Generation")

DATASET_DIR = BASE_DIR / "dataset"

OUTPUT_DIR = BASE_DIR / "processed_data" / "hum"

CSV_PATH = DATASET_DIR / "MLEndHWD_Audio_Attributes.csv"

TARGET_SR = 22050

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================
# LOAD METADATA
# ==================================

df = pd.read_csv(CSV_PATH)

hum_df = df[df["Interpretation"] == "Hum"]

print("=" * 50)
print("AUDIO PREPROCESSING")
print("=" * 50)

print("Total Hum Recordings:", len(hum_df))
print("Target Sample Rate:", TARGET_SR)

# ==================================
# FIND AUDIO FILES
# ==================================

audio_lookup = {
    file.name.lower(): file
    for file in DATASET_DIR.rglob("*.wav")
}

processed = 0
failed = []

# ==================================
# PROCESS AUDIO
# ==================================

for _, row in tqdm(hum_df.iterrows(), total=len(hum_df)):

    filename = Path(str(row["Public filename"])).name

    input_path = audio_lookup.get(filename.lower())

    if input_path is None:
        failed.append(filename)
        continue

    try:

        # Load audio as mono and resample
        audio, sr = librosa.load(
            input_path,
            sr=TARGET_SR,
            mono=True
        )

        # Peak normalization
        peak = np.max(np.abs(audio))

        if peak > 0:
            audio = audio / peak

        # Save processed audio
        output_path = OUTPUT_DIR / filename

        sf.write(
            output_path,
            audio,
            TARGET_SR,
            subtype="PCM_16"
        )

        processed += 1

    except Exception as e:

        failed.append(filename)

        print(f"\nFailed: {filename}")
        print("Error:", e)

# ==================================
# FINAL REPORT
# ==================================

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETED")
print("=" * 50)

print("Total Hum Recordings:", len(hum_df))
print("Successfully Processed:", processed)
print("Failed Files:", len(failed))

print("Output Directory:", OUTPUT_DIR)

# Save failed files
pd.DataFrame({"Failed Files": failed}).to_csv(
    BASE_DIR / "documentation" / "preprocessing_failures.csv",
    index=False
)

print("\nFailure report saved.")