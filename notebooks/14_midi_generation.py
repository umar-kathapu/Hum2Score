import pandas as pd
import pretty_midi
from pathlib import Path

# Project paths
BASE_DIR = Path(r"D:\Musical-Score-Generation")

INPUT_FILE = (
    BASE_DIR / "outputs" / "note_segmentation"
    / "smoothed_notes_0000.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "midi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load smoothed notes
df = pd.read_csv(INPUT_FILE)

# Create MIDI object
midi = pretty_midi.PrettyMIDI(
    initial_tempo=120
)

# Create instrument
instrument = pretty_midi.Instrument(
    program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
)

# Convert each note into MIDI format
for _, row in df.iterrows():

    note_name = row["Note"]

    start_time = float(row["Start Time"])
    end_time = float(row["End Time"])

    # Convert note name to MIDI pitch
    note_name = note_name.replace("♯", "#")
    
    midi_pitch = pretty_midi.note_name_to_number(note_name)

    note = pretty_midi.Note(
        velocity=100,
        pitch=midi_pitch,
        start=start_time,
        end=end_time
    )

    instrument.notes.append(note)

# Add instrument
midi.instruments.append(instrument)

# Save MIDI file
output_file = OUTPUT_DIR / "generated_music_0000.mid"

midi.write(str(output_file))

print("\nMIDI GENERATION COMPLETED!")
print("Total Notes:", len(instrument.notes))
print("MIDI File:", output_file)