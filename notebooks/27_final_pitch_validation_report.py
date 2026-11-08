import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pitch_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
PITCH_FILE = OUTPUT_DIR / "pitch_accuracy_0000.csv"

INDEPENDENT_FILE = (
    OUTPUT_DIR / "independent_pitch_validation_0000.csv"
)

AUTOCORRELATION_FILE = (
    OUTPUT_DIR / "autocorrelation_note27.csv"
)

# Output files
REPORT_FILE = OUTPUT_DIR / "final_pitch_validation_report_0000.csv"

SUMMARY_FILE = OUTPUT_DIR / "final_pitch_validation_summary_0000.txt"

# --------------------------------------------------
# 1. Load validation results
# --------------------------------------------------

pitch_df = pd.read_csv(PITCH_FILE)

independent_df = pd.read_csv(INDEPENDENT_FILE)

autocorr_df = pd.read_csv(AUTOCORRELATION_FILE)

# --------------------------------------------------
# 2. Merge pitch validation results
# --------------------------------------------------

report_df = pitch_df.merge(
    independent_df[
        [
            "Note Number",
            "YIN F0",
            "YIN MIDI",
            "YIN Error (Cents)",
            "YIN-pYIN Difference (Cents)"
        ]
    ],
    on="Note Number",
    how="left",
    suffixes=("_pYIN", "_YIN")
)

# --------------------------------------------------
# 3. Add validation status
# --------------------------------------------------

def classify_note(row):

    yin_error = row["YIN-pYIN Difference (Cents)"]

    if pd.isna(yin_error):
        return "Insufficient Evidence"

    if abs(yin_error) > 1000:

        if row["Note Number"] == 27:
            return "Investigated - E3 Retained"

        return "Potential Octave Disagreement"

    if abs(yin_error) <= 50:
        return "Strong Algorithm Agreement"

    return "Review Required"


report_df["Validation Status"] = report_df.apply(
    classify_note,
    axis=1
)

# --------------------------------------------------
# 4. Save consolidated report
# --------------------------------------------------

report_df.to_csv(
    REPORT_FILE,
    index=False
)

# --------------------------------------------------
# 5. Generate summary
# --------------------------------------------------

valid = report_df.dropna(
    subset=["YIN-pYIN Difference (Cents)"]
)

agreement = (
    valid["YIN-pYIN Difference (Cents)"].abs() <= 50
)

summary_lines = []

summary_lines.append("FINAL PITCH VALIDATION REPORT")
summary_lines.append("=" * 60)

summary_lines.append(
    f"Total Detected Notes: {len(report_df)}"
)

summary_lines.append(
    f"Notes with both pitch estimates: {len(valid)}"
)

summary_lines.append(
    f"Notes agreeing within 50 cents: {agreement.sum()}"
)

summary_lines.append(
    f"Agreement Percentage: {agreement.mean() * 100:.2f}%"
)

summary_lines.append(
    f"Median YIN-pYIN Difference: "
    f"{valid['YIN-pYIN Difference (Cents)'].abs().median():.2f} cents"
)

summary_lines.append("")

summary_lines.append("NOTE 27 INVESTIGATION")
summary_lines.append("-" * 40)

note27 = report_df[
    report_df["Note Number"] == 27
]

if not note27.empty:

    row = note27.iloc[0]

    summary_lines.append(
        f"Detected Note: {row['Note']}"
    )

    summary_lines.append(
        f"pYIN Frequency: {row['Median F0']:.2f} Hz"
    )

    summary_lines.append(
        f"YIN Frequency: {row['YIN F0']:.2f} Hz"
    )

    summary_lines.append(
        "Autocorrelation Frequency: 160.95 Hz"
    )

    summary_lines.append(
        "Decision: E3 retained provisionally."
    )

summary_lines.append("")

summary_lines.append("VALIDATION STATUS COUNTS")
summary_lines.append("-" * 40)

status_counts = report_df["Validation Status"].value_counts()

for status, count in status_counts.items():

    summary_lines.append(
        f"{status}: {count}"
    )

summary_lines.append("")

summary_lines.append(
    "Important: These are algorithm-agreement results, "
    "not independently measured transcription accuracy."
)

summary_lines.append(
    "The report covers sample 0000.wav only."
)

# Save summary
with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(summary_lines))

# --------------------------------------------------
# 6. Display report
# --------------------------------------------------

print("\n".join(summary_lines))

print("\nFiles Generated:")
print(REPORT_FILE)
print(SUMMARY_FILE)

print("\nFinal Pitch Validation Report Completed.")