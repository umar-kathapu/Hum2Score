import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OLD_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "note_segmentation"
    / "smoothed_notes_0000.csv"
)

NEW_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "improved_segmentation"
    / "improved_notes_0000.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "segmentation_comparison_0000.csv"

PLOT_OUTPUT = OUTPUT_DIR / "segmentation_comparison_0000.png"


# --------------------------------------------------
# 2. Load CSV files
# --------------------------------------------------

old_df = pd.read_csv(OLD_FILE)
new_df = pd.read_csv(NEW_FILE)


# --------------------------------------------------
# 3. Standardize column names
# --------------------------------------------------

def prepare_dataframe(df, source_name):

    df = df.copy()

    # Identify columns
    start_candidates = [
        "Start Time", "start_time", "Start", "start"
    ]

    end_candidates = [
        "End Time", "end_time", "End", "end"
    ]

    note_candidates = [
        "Note", "note", "Note Name", "Note_Name"
    ]

    def find_column(candidates):
        return next(
            (col for col in candidates if col in df.columns),
            None
        )

    start_col = find_column(start_candidates)
    end_col = find_column(end_candidates)
    note_col = find_column(note_candidates)

    if not all([start_col, end_col, note_col]):

        raise ValueError(
            f"{source_name}: Required columns missing. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            start_col: "Start Time",
            end_col: "End Time",
            note_col: "Note"
        }
    )

    df["Start Time"] = pd.to_numeric(
        df["Start Time"], errors="coerce"
    )

    df["End Time"] = pd.to_numeric(
        df["End Time"], errors="coerce"
    )

    df = df.dropna(
        subset=["Start Time", "End Time", "Note"]
    )

    df = df.sort_values(
        "Start Time"
    ).reset_index(drop=True)

    if (df["End Time"] <= df["Start Time"]).any():
        raise ValueError(
            f"{source_name}: Invalid note boundaries found."
        )

    df["Duration"] = (
        df["End Time"] - df["Start Time"]
    )

    # Normalize sharp notation
    df["Note"] = (
        df["Note"]
        .astype(str)
        .str.replace("♯", "#", regex=False)
    )

    return df[
        ["Start Time", "End Time", "Duration", "Note"]
    ]


old_df = prepare_dataframe(old_df, "Old segmentation")
new_df = prepare_dataframe(new_df, "Improved segmentation")


# --------------------------------------------------
# 4. Match notes based on temporal overlap
# --------------------------------------------------

# A pair is considered a candidate match when
# its temporal overlap is at least 50% of the
# shorter segment.

candidates = []

for old_index, old_row in old_df.iterrows():

    for new_index, new_row in new_df.iterrows():

        overlap_start = max(
            old_row["Start Time"],
            new_row["Start Time"]
        )

        overlap_end = min(
            old_row["End Time"],
            new_row["End Time"]
        )

        overlap = max(
            0,
            overlap_end - overlap_start
        )

        shorter_duration = min(
            old_row["Duration"],
            new_row["Duration"]
        )

        overlap_ratio = (
            overlap / shorter_duration
            if shorter_duration > 0 else 0
        )

        if overlap_ratio >= 0.50:

            candidates.append({
                "Old Index": old_index,
                "New Index": new_index,
                "Overlap Duration": overlap,
                "Overlap Ratio": overlap_ratio,
                "Old Note": old_row["Note"],
                "New Note": new_row["Note"],
                "Same Pitch": (
                    old_row["Note"] == new_row["Note"]
                ),
                "Old Start": old_row["Start Time"],
                "New Start": new_row["Start Time"],
                "Start Difference": abs(
                    old_row["Start Time"]
                    - new_row["Start Time"]
                )
            })


# Greedy one-to-one matching:
# Select strongest overlap first and ensure
# each old/new segment is matched only once.

candidates.sort(
    key=lambda x: (
        x["Overlap Ratio"],
        x["Overlap Duration"]
    ),
    reverse=True
)

matched_old = set()
matched_new = set()

matches = []

for candidate in candidates:

    old_index = candidate["Old Index"]
    new_index = candidate["New Index"]

    if (
        old_index not in matched_old
        and new_index not in matched_new
    ):

        matched_old.add(old_index)
        matched_new.add(new_index)

        matches.append(candidate)


# --------------------------------------------------
# 5. Create comparison report
# --------------------------------------------------

comparison_df = pd.DataFrame(matches)

if not comparison_df.empty:

    comparison_df["Pitch Match"] = np.where(
        comparison_df["Same Pitch"],
        "Same Pitch",
        "Different Pitch"
    )

    comparison_df["Boundary Difference (sec)"] = (
        comparison_df["Start Difference"]
    )

comparison_df.to_csv(
    CSV_OUTPUT,
    index=False
)


# --------------------------------------------------
# 6. Identify unmatched segments
# --------------------------------------------------

unmatched_old = old_df.loc[
    ~old_df.index.isin(matched_old)
].copy()

unmatched_new = new_df.loc[
    ~new_df.index.isin(matched_new)
].copy()


# --------------------------------------------------
# 7. Print statistics
# --------------------------------------------------

same_pitch_count = (
    comparison_df["Same Pitch"].sum()
    if not comparison_df.empty else 0
)

different_pitch_count = (
    len(comparison_df) - same_pitch_count
)

print("\nSEGMENTATION COMPARISON")
print("=" * 65)

print(f"Old Segmentation Notes: {len(old_df)}")
print(f"Improved Segmentation Notes: {len(new_df)}")

print(f"\nMatched Note Pairs: {len(matches)}")
print(f"Same Pitch Matches: {same_pitch_count}")
print(f"Different Pitch Matches: {different_pitch_count}")

print(f"Unmatched Old Notes: {len(unmatched_old)}")
print(f"Unmatched Improved Notes: {len(unmatched_new)}")

if not comparison_df.empty:

    print(
        "\nMean Start-Time Difference: "
        f"{comparison_df['Start Difference'].mean():.4f} sec"
    )

    print(
        "Median Start-Time Difference: "
        f"{comparison_df['Start Difference'].median():.4f} sec"
    )

    print(
        "Mean Temporal Overlap: "
        f"{comparison_df['Overlap Ratio'].mean() * 100:.2f}%"
    )


# --------------------------------------------------
# 8. Display unmatched notes
# --------------------------------------------------

print("\nUNMATCHED OLD NOTES")
print("-" * 45)

if unmatched_old.empty:
    print("None")
else:
    print(
        unmatched_old.to_string(index=True)
    )


print("\nUNMATCHED IMPROVED NOTES")
print("-" * 45)

if unmatched_new.empty:
    print("None")
else:
    print(
        unmatched_new.to_string(index=True)
    )


# --------------------------------------------------
# 9. Visualization
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(16, 8),
    sharex=True
)


# Old segmentation
for _, row in old_df.iterrows():

    axes[0].barh(
        0,
        row["Duration"],
        left=row["Start Time"],
        height=0.5,
        edgecolor="black"
    )

    axes[0].text(
        row["Start Time"] + row["Duration"] / 2,
        0,
        row["Note"],
        ha="center",
        va="center",
        fontsize=7
    )


# Improved segmentation
for _, row in new_df.iterrows():

    axes[1].barh(
        0,
        row["Duration"],
        left=row["Start Time"],
        height=0.5,
        edgecolor="black"
    )

    axes[1].text(
        row["Start Time"] + row["Duration"] / 2,
        0,
        row["Note"],
        ha="center",
        va="center",
        fontsize=7
    )


axes[0].set_title(
    f"Original Segmentation ({len(old_df)} Notes)"
)

axes[1].set_title(
    f"Improved Segmentation ({len(new_df)} Notes)"
)

for ax in axes:

    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3)

axes[1].set_xlabel("Time (seconds)")

plt.tight_layout()

plt.savefig(
    PLOT_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 10. Final output
# --------------------------------------------------

print("\nFILES SAVED")
print("=" * 45)

print(CSV_OUTPUT)
print(PLOT_OUTPUT)

print("\nSegmentation Comparison Completed.")