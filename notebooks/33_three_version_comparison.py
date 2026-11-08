import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "Original V1 (47)": (
        ROOT / "outputs" / "note_segmentation"
        / "smoothed_notes_0000.csv"
    ),

    "Improved V2 (50)": (
        ROOT / "outputs" / "improved_segmentation"
        / "improved_notes_0000.csv"
    ),

    "Stable V3 (49)": (
        ROOT / "outputs" / "stable_segmentation"
        / "stable_notes_0000.csv"
    )
}

OUTPUT_DIR = ROOT / "outputs" / "three_version_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and standardize CSV files
# --------------------------------------------------

def load_notes(path, version):

    if not path.exists():
        raise FileNotFoundError(
            f"{version} file not found: {path}"
        )

    df = pd.read_csv(path)

    # Normalize column names for matching
    normalized = {
        str(col).strip().lower().replace("_", " "): col
        for col in df.columns
    }

    def find_column(candidates):

        for candidate in candidates:
            if candidate in normalized:
                return normalized[candidate]

        return None

    start_col = find_column([
        "start time", "start", "onset"
    ])

    end_col = find_column([
        "end time", "end", "offset"
    ])

    note_col = find_column([
        "note", "note name", "pitch name"
    ])

    midi_col = find_column([
        "midi", "midi note", "midi pitch"
    ])

    if start_col is None or end_col is None:
        raise ValueError(
            f"{version}: Start/End columns not found. "
            f"Available columns: {list(df.columns)}"
        )

    if note_col is None and midi_col is None:
        raise ValueError(
            f"{version}: Neither Note nor MIDI column found."
        )

    result = pd.DataFrame()

    result["Start"] = pd.to_numeric(
        df[start_col], errors="coerce"
    )

    result["End"] = pd.to_numeric(
        df[end_col], errors="coerce"
    )

    if midi_col is not None:

        result["MIDI"] = pd.to_numeric(
            df[midi_col], errors="coerce"
        )

    else:

        note_names = (
            df[note_col]
            .astype(str)
            .str.replace("♯", "#", regex=False)
            .str.replace("♭", "b", regex=False)
        )

        result["MIDI"] = note_names.apply(
            lambda x: (
                __import__("librosa").note_to_midi(x)
                if x.lower() != "nan"
                else np.nan
            )
        )

    result = result.dropna(
        subset=["Start", "End", "MIDI"]
    ).copy()

    result["MIDI"] = result["MIDI"].round().astype(int)

    result = result[
        (result["End"] > result["Start"])
    ]

    result = result.sort_values(
        "Start"
    ).reset_index(drop=True)

    result["Duration"] = result["End"] - result["Start"]

    result["Note"] = result["MIDI"].apply(
        lambda x: __import__("librosa").midi_to_note(int(x))
    )

    result["Version"] = version

    return result[
        ["Version", "Start", "End", "Duration", "MIDI", "Note"]
    ]


versions = {
    name: load_notes(path, name)
    for name, path in FILES.items()
}


# --------------------------------------------------
# 3. Print individual statistics
# --------------------------------------------------

print("\nTHREE-VERSION SEGMENTATION COMPARISON")
print("=" * 70)

summary = []

for name, df in versions.items():

    summary.append({
        "Version": name,
        "Total Notes": len(df),
        "Average Duration": df["Duration"].mean(),
        "Median Duration": df["Duration"].median(),
        "Shortest Note": df["Duration"].min(),
        "Longest Note": df["Duration"].max()
    })

summary_df = pd.DataFrame(summary)

print("\nVERSION STATISTICS")
print("-" * 70)

print(summary_df.to_string(index=False))


# --------------------------------------------------
# 4. Pairwise temporal matching
# --------------------------------------------------

def compare_versions(df_a, df_b, name_a, name_b):

    candidates = []

    for i, a in df_a.iterrows():

        for j, b in df_b.iterrows():

            overlap_start = max(a["Start"], b["Start"])
            overlap_end = min(a["End"], b["End"])

            overlap = max(
                0,
                overlap_end - overlap_start
            )

            shorter = min(
                a["Duration"],
                b["Duration"]
            )

            overlap_ratio = (
                overlap / shorter
                if shorter > 0 else 0
            )

            if overlap_ratio >= 0.50:

                candidates.append({
                    "Index A": i,
                    "Index B": j,
                    "Overlap Ratio": overlap_ratio,
                    "Overlap Duration": overlap,
                    "Note A": a["Note"],
                    "Note B": b["Note"],
                    "Same Pitch": a["MIDI"] == b["MIDI"],
                    "Start Difference": abs(
                        a["Start"] - b["Start"]
                    ),
                    "Duration Difference": abs(
                        a["Duration"] - b["Duration"]
                    )
                })

    candidates.sort(
        key=lambda x: (
            x["Overlap Ratio"],
            x["Overlap Duration"]
        ),
        reverse=True
    )

    used_a = set()
    used_b = set()
    matches = []

    for item in candidates:

        i = item["Index A"]
        j = item["Index B"]

        if i not in used_a and j not in used_b:

            used_a.add(i)
            used_b.add(j)

            matches.append(item)

    unmatched_a = len(df_a) - len(used_a)
    unmatched_b = len(df_b) - len(used_b)

    same_pitch = sum(
        item["Same Pitch"]
        for item in matches
    )

    different_pitch = len(matches) - same_pitch

    result = {
        "Comparison": f"{name_a} vs {name_b}",
        "Notes A": len(df_a),
        "Notes B": len(df_b),
        "Matched Pairs": len(matches),
        "Same Pitch": same_pitch,
        "Different Pitch": different_pitch,
        "Unmatched A": unmatched_a,
        "Unmatched B": unmatched_b,
        "Mean Overlap (%)": (
            np.mean([
                item["Overlap Ratio"]
                for item in matches
            ]) * 100
            if matches else 0
        ),
        "Mean Start Difference (sec)": (
            np.mean([
                item["Start Difference"]
                for item in matches
            ])
            if matches else np.nan
        )
    }

    details = pd.DataFrame(matches)

    return result, details


# --------------------------------------------------
# 5. Run all three pairwise comparisons
# --------------------------------------------------

names = list(versions.keys())

pairwise_results = []
pairwise_details = {}

for i in range(len(names)):

    for j in range(i + 1, len(names)):

        name_a = names[i]
        name_b = names[j]

        result, details = compare_versions(
            versions[name_a],
            versions[name_b],
            name_a,
            name_b
        )

        pairwise_results.append(result)

        pairwise_details[
            f"{i+1}_{j+1}"
        ] = details


pairwise_df = pd.DataFrame(pairwise_results)

print("\nPAIRWISE COMPARISON")
print("-" * 70)

print(
    pairwise_df.to_string(index=False)
)


# --------------------------------------------------
# 6. Save all reports
# --------------------------------------------------

summary_df.to_csv(
    OUTPUT_DIR / "version_statistics.csv",
    index=False
)

pairwise_df.to_csv(
    OUTPUT_DIR / "pairwise_comparison_summary.csv",
    index=False
)

for key, details in pairwise_details.items():

    details.to_csv(
        OUTPUT_DIR / f"pairwise_details_{key}.csv",
        index=False
    )


# --------------------------------------------------
# 7. Plot all three note sequences
# --------------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(17, 10),
    sharex=True
)

for ax, (name, df) in zip(
    axes,
    versions.items()
):

    for _, row in df.iterrows():

        ax.barh(
            0,
            row["Duration"],
            left=row["Start"],
            height=0.5,
            edgecolor="black"
        )

        ax.text(
            row["Start"] + row["Duration"] / 2,
            0,
            row["Note"],
            ha="center",
            va="center",
            fontsize=7
        )

    ax.set_title(
        f"{name} - {len(df)} Notes"
    )

    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3)


axes[-1].set_xlabel("Time (seconds)")

plt.tight_layout()

plot_file = (
    OUTPUT_DIR / "three_version_comparison_0000.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 8. Final report
# --------------------------------------------------

print("\nFILES GENERATED")
print("=" * 60)

print(OUTPUT_DIR)

print("\nThree-Version Comparison Completed.")