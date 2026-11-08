import librosa
import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(r"D:\Musical-Score-Generation")
AUDIO_DIR = BASE_DIR / "processed_data" / "hum"
OUTPUT_DIR = BASE_DIR / "outputs" / "note_segmentation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Select audio file
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

# Time calculation
times = librosa.times_like(f0, sr=sr, hop_length=256)

# Convert detected frequencies into MIDI notes
midi_notes = np.full(len(f0), np.nan)

midi_notes[voiced_flag] = librosa.hz_to_midi(f0[voiced_flag])

# Round MIDI values to nearest semitone
rounded_notes = np.full(len(f0), np.nan)
rounded_notes[voiced_flag] = np.round(midi_notes[voiced_flag])

# Convert MIDI notes to note names
note_names = np.full(len(f0), None, dtype=object)

note_names[voiced_flag] = librosa.midi_to_note(
    rounded_notes[voiced_flag].astype(int)
)

# Segment consecutive identical notes
segments = []

current_note = None
start_time = None
previous_time = None

for i in range(len(note_names)):

    note = note_names[i]
    current_time = times[i]

    if note != current_note:

        if current_note is not None:
            segments.append({
                "Note": current_note,
                "Start Time": start_time,
                "End Time": previous_time + 256 / sr,
                "Duration": previous_time + 256 / sr - start_time
            })

        current_note = note

        if note is not None:
            start_time = current_time
        else:
            start_time = None

    previous_time = current_time

# Save final segment
if current_note is not None:
    segments.append({
        "Note": current_note,
        "Start Time": start_time,
        "End Time": previous_time + 256 / sr,
        "Duration": previous_time + 256 / sr - start_time
    })

# Create DataFrame
df = pd.DataFrame(segments)

# Remove very short segments (less than 0.08 seconds)
df = df[df["Duration"] >= 0.08].copy()

# Save CSV
output_file = OUTPUT_DIR / "notes_0000.csv"
df.to_csv(output_file, index=False)

# Display results
print("\nNOTE SEGMENTATION COMPLETED!")
print("\nDetected Musical Notes:")
print(df.to_string(index=False))

print("\nTotal Segments:", len(df))
print("Output saved to:", output_file)