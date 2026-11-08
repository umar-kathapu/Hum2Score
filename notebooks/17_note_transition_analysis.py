import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(r"D:\Musical-Score-Generation")

INPUT_FILE = (
    BASE_DIR / "outputs" / "note_segmentation"
    / "smoothed_notes_0000.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "pitch_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD NOTE DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

df = df.sort_values("Start Time").reset_index(drop=True)

print("Total Notes:", len(df))

# ==========================================
# 3. NOTE TO MIDI CONVERSION
# ==========================================

def note_to_midi(note_name):

    try:
        return int(
            __import__("librosa").note_to_midi(
                str(note_name).replace("♯", "#")
            )
        )

    except Exception:
        return np.nan


df["MIDI"] = df["Note"].apply(note_to_midi)

# ==========================================
# 4. ANALYSE CONSECUTIVE NOTES
# ==========================================

results = []

for i in range(len(df) - 1):

    current = df.iloc[i]
    following = df.iloc[i + 1]

    current_end = float(current["End Time"])
    next_start = float(following["Start Time"])

    gap = next_start - current_end

    pitch_difference = abs(
        current["MIDI"] - following["MIDI"]
    )

    same_pitch = (
        current["MIDI"] == following["MIDI"]
    )

    results.append({

        "Current Note": current["Note"],

        "Next Note": following["Note"],

        "Current Start": current["Start Time"],

        "Current End": current["End Time"],

        "Next Start": following["Start Time"],

        "Next End": following["End Time"],

        "Current Duration": current["Duration"],

        "Next Duration": following["Duration"],

        "Gap": gap,

        "Pitch Difference (Semitones)": pitch_difference,

        "Same Pitch": same_pitch,

        "Potential Merge": (
            same_pitch
            and -0.03 <= gap <= 0.08
        )
    })

# ==========================================
# 5. CREATE ANALYSIS REPORT
# ==========================================

analysis_df = pd.DataFrame(results)

output_file = (
    OUTPUT_DIR / "note_transition_analysis.csv"
)

analysis_df.to_csv(output_file, index=False)

# ==========================================
# 6. DISPLAY SUMMARY
# ==========================================

print("\n====================================")
print("NOTE TRANSITION ANALYSIS COMPLETED")
print("====================================")

print("Total Note Transitions:", len(analysis_df))

print(
    "Same Pitch Transitions:",
    int(analysis_df["Same Pitch"].sum())
)

print(
    "Potential Merge Candidates:",
    int(analysis_df["Potential Merge"].sum())
)

print(
    "Average Inter-note Gap:",
    round(analysis_df["Gap"].mean(), 4),
    "seconds"
)

print("\nPotential Merge Candidates:")

print(
    analysis_df[
        analysis_df["Potential Merge"]
    ][
        [
            "Current Note",
            "Next Note",
            "Gap",
            "Current Duration",
            "Next Duration"
        ]
    ].to_string(index=False)
)

print("\nFull report saved at:")
print(output_file)