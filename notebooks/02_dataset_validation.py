import pandas as pd
from pathlib import Path
from collections import Counter

# Dataset location
BASE_DIR = Path(r"D:\Musical-Score-Generation\dataset")

CSV_PATH = BASE_DIR / "MLEndHWD_Audio_Attributes.csv"

# Load metadata
df = pd.read_csv(CSV_PATH)

# Get audio filenames
csv_files = df["Public filename"].astype(str).str.strip()

# Find actual WAV files
wav_files = list(BASE_DIR.rglob("*.wav"))

# Create filename lookup
audio_lookup = {
    file.name.lower(): file
    for file in wav_files
}

# Validation
matched = []
missing = []
duplicates = []

print("=" * 55)
print("DATASET VALIDATION")
print("=" * 55)

print("\n1. CSV RECORDS:", len(df))
print("2. ACTUAL WAV FILES:", len(wav_files))

# Check filename matching
for filename in csv_files:

    key = Path(filename).name.lower()

    if key in audio_lookup:
        matched.append(filename)
    else:
        missing.append(filename)

# Check duplicate filenames in CSV
duplicate_names = csv_files[csv_files.duplicated()]

# Check duplicate records
print("\n3. MATCHING RESULTS")
print("Matched Files:", len(matched))
print("Missing Files:", len(missing))
print("Duplicate CSV Filenames:", len(duplicate_names))

# Check duplicate audio file names
name_counts = Counter(file.name.lower() for file in wav_files)

duplicate_audio_names = {
    name: count
    for name, count in name_counts.items()
    if count > 1
}

print("Duplicate Audio Filenames:", len(duplicate_audio_names))

# Check Hum and Whistle
print("\n4. RECORDING DISTRIBUTION")
print(df["Interpretation"].value_counts())

# Check missing metadata
print("\n5. MISSING METADATA")
print(df.isnull().sum())

# Save validation report
report = pd.DataFrame({
    "Metric": [
        "CSV Records",
        "WAV Files",
        "Matched Files",
        "Missing Files",
        "Duplicate CSV Filenames",
        "Duplicate Audio Filenames"
    ],
    "Count": [
        len(df),
        len(wav_files),
        len(matched),
        len(missing),
        len(duplicate_names),
        len(duplicate_audio_names)
    ]
})

report.to_csv(
    BASE_DIR / "dataset_validation_report.csv",
    index=False
)

print("\nValidation report saved successfully!")
print("\nDATASET VALIDATION COMPLETED!")