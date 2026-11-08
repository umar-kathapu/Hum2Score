import pandas as pd
from music21 import stream, note, meter, clef, tempo, metadata
from music21 import duration
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(r"D:\Musical-Score-Generation")

INPUT_FILE = (
    BASE_DIR / "outputs" / "note_segmentation"
    / "smoothed_notes_0000.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "musicxml"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

df = df.sort_values("Start Time").reset_index(drop=True)

print("Total detected notes:", len(df))

# ==========================================
# 3. STANDARD MUSICAL DURATIONS
# ==========================================

# Values are measured in quarter-note units.
VALID_DURATIONS = [
    4.0,    # Whole note
    3.0,    # Dotted half note
    2.0,    # Half note
    1.5,    # Dotted quarter note
    1.0,    # Quarter note
    0.75,   # Dotted eighth note
    0.5,    # Eighth note
    0.375,  # Dotted sixteenth note
    0.25    # Sixteenth note
]

def quantize_duration(value):

    return min(
        VALID_DURATIONS,
        key=lambda x: abs(x - value)
    )


def add_rests(part, rest_length):

    remaining = rest_length

    while remaining >= 0.25:

        available = [
            d for d in VALID_DURATIONS
            if d <= remaining + 0.000001
        ]

        if not available:
            break

        rest_duration = max(available)

        rest = note.Rest()
        rest.duration = duration.Duration(rest_duration)

        part.append(rest)

        remaining -= rest_duration


# ==========================================
# 4. CREATE SCORE
# ==========================================

score = stream.Score()

score_metadata = metadata.Metadata()
score_metadata.title = "Generated Music from Humming"
score_metadata.composer = "AI Musical Score Generation"

score.insert(0, score_metadata)

part = stream.Part()

part.insert(0, clef.TrebleClef())
part.insert(0, meter.TimeSignature("4/4"))
part.insert(0, tempo.MetronomeMark(number=161.5))

# ==========================================
# 5. GENERATE NOTES AND RESTS
# ==========================================

previous_end = 0.0

for _, row in df.iterrows():

    note_name = str(row["Note"]).replace("♯", "#")

    start_time = float(row["Start Time"])
    end_time = float(row["End Time"])

    original_duration = end_time - start_time

    # Convert seconds into approximate quarter-note units.
    # This uses the estimated tempo as a provisional scale.
    seconds_per_beat = 60 / 161.5

    target_start = start_time / seconds_per_beat
    target_end = end_time / seconds_per_beat

    # Insert silence only when a positive gap exists.
    gap_seconds = max(0.0, start_time - previous_end)

    if gap_seconds > 0.02:

        gap_beats = gap_seconds / seconds_per_beat

        add_rests(part, gap_beats)

    # Quantize note duration
    target_duration = max(
        0.25,
        (target_end - target_start)
    )

    final_duration = quantize_duration(target_duration)

    # Create note
    musical_note = note.Note(note_name)

    musical_note.duration = duration.Duration(final_duration)

    part.append(musical_note)

    previous_end = max(previous_end, end_time)

# ==========================================
# 6. ADD PART TO SCORE
# ==========================================

score.append(part)

# ==========================================
# 7. EXPORT MUSICXML
# ==========================================

output_file = OUTPUT_DIR / "generated_score_0000.musicxml"

score.write("musicxml", fp=str(output_file))

# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n================================")
print("MUSICXML GENERATION COMPLETED!")
print("================================")

print("Total detected notes:", len(df))
print("Time Signature: 4/4")
print("Estimated Tempo: 161.5 BPM")

print("\nOutput File:")
print(output_file)

print("\nOpen the file using MuseScore.")
