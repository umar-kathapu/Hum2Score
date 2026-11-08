import librosa
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(r"D:\Musical-Score-Generation")

AUDIO_FILE = (
    BASE_DIR / "processed_data" / "hum" / "0000.wav"
)

SEGMENT_FILE = (
    BASE_DIR / "outputs" / "note_segmentation"
    / "smoothed_notes_0000.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "pitch_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD AUDIO
# ==========================================

y, sr = librosa.load(AUDIO_FILE, sr=22050)

print("Audio loaded successfully!")
print("Sample Rate:", sr)
print("Duration:", round(len(y) / sr, 2), "seconds")

# ==========================================
# 3. FRESH PITCH DETECTION
# ==========================================

f0, voiced_flag, voiced_prob = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr,
    frame_length=2048,
    hop_length=256
)

times = librosa.times_like(
    f0,
    sr=sr,
    hop_length=256
)

# ==========================================
# 4. CONVERT FREQUENCY TO MIDI
# ==========================================

valid_midi = librosa.hz_to_midi(f0[voiced_flag])

rounded_midi = np.round(valid_midi)

# ==========================================
# 5. PITCH STABILITY ANALYSIS
# ==========================================

midi_changes = np.diff(valid_midi)

pitch_std = np.std(valid_midi)

stable_frames = np.sum(
    np.abs(midi_changes) < 0.5
)

total_changes = len(midi_changes)

stability_percentage = (
    stable_frames / total_changes * 100
    if total_changes > 0 else 0
)

# ==========================================
# 6. OCTAVE JUMP DETECTION
# ==========================================

octave_jumps = np.where(
    np.abs(midi_changes) >= 11
)[0]

# ==========================================
# 7. NOTE DISTRIBUTION
# ==========================================

note_names = librosa.midi_to_note(
    rounded_midi.astype(int)
)

note_distribution = pd.Series(
    note_names
).value_counts()

# ==========================================
# 8. SAVE FRAME-LEVEL RESULTS
# ==========================================

pitch_df = pd.DataFrame({
    "Time": times[voiced_flag],
    "Frequency": f0[voiced_flag],
    "MIDI": valid_midi,
    "Detected Note": note_names,
    "Voiced Probability": voiced_prob[voiced_flag]
})

pitch_df.to_csv(
    OUTPUT_DIR / "pitch_validation_frames.csv",
    index=False
)

# ==========================================
# 9. SAVE OCTAVE JUMP RESULTS
# ==========================================

jump_data = []

for index in octave_jumps:

    jump_data.append({
        "Time": times[voiced_flag][index + 1],
        "Previous MIDI": valid_midi[index],
        "Current MIDI": valid_midi[index + 1],
        "MIDI Difference": midi_changes[index]
    })

jump_df = pd.DataFrame(jump_data)

jump_df.to_csv(
    OUTPUT_DIR / "octave_jump_report.csv",
    index=False
)

# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("\n================================")
print("PITCH VALIDATION COMPLETED")
print("================================")

print("Total Voiced Frames:", len(valid_midi))

print(
    "Pitch Standard Deviation:",
    round(float(pitch_std), 3)
)

print(
    "Pitch Stability Percentage:",
    round(float(stability_percentage), 2), "%"
)

print("Potential Large Pitch Jumps:", len(octave_jumps))

print("\nMost Frequent Detected Notes:")
print(note_distribution.head(15))

print("\nPotential Large Pitch Jumps:")
print(jump_df.to_string(index=False))

print("\nReports saved at:")
print(OUTPUT_DIR)