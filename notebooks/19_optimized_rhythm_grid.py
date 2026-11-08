import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "note_segmentation"
    / "smoothed_notes_0000.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rhythm_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "optimized_rhythm_grid_0000.csv"

# Load note data
df = pd.read_csv(INPUT_FILE)

# Identify onset column
possible_columns = ["Start Time", "start_time", "Start", "start"]

onset_column = next(
    (col for col in possible_columns if col in df.columns),
    None
)

if onset_column is None:
    raise ValueError(
        f"Start-time column not found. Available columns: {list(df.columns)}"
    )

onsets = pd.to_numeric(
    df[onset_column],
    errors="coerce"
).dropna().to_numpy()

if len(onsets) < 2:
    raise ValueError("Not enough valid note onsets for rhythm analysis.")

# Candidate grid durations in seconds
grid_durations = [
    0.0625,
    0.09375,
    0.125,
    0.1875,
    0.25,
    0.375,
    0.5
]

# Fixed alignment tolerances
tolerance_30ms = 0.030
tolerance_50ms = 0.050

results = []

for grid in grid_durations:

    best_phase = 0
    best_mean_error = float("inf")
    best_errors = None

    # Test different starting offsets
    phases = np.arange(0, grid, 0.001)

    for phase in phases:

        # Distance from each onset to its nearest grid point
        relative_positions = (onsets - phase) / grid

        nearest_grid_positions = np.round(relative_positions) * grid

        aligned_onsets = phase + nearest_grid_positions

        errors = np.abs(onsets - aligned_onsets)

        mean_error = np.mean(errors)

        if mean_error < best_mean_error:
            best_mean_error = mean_error
            best_phase = phase
            best_errors = errors

    results.append({
        "Grid Duration": grid,
        "Best Phase Offset": best_phase,
        "Mean Onset Error": np.mean(best_errors),
        "Median Onset Error": np.median(best_errors),
        "90th Percentile Error": np.percentile(best_errors, 90),
        "Alignment Within 30ms (%)": (
            np.mean(best_errors <= tolerance_30ms) * 100
        ),
        "Alignment Within 50ms (%)": (
            np.mean(best_errors <= tolerance_50ms) * 100
        )
    })

# Create results table
results_df = pd.DataFrame(results)

# Sort by mean onset error
results_df = results_df.sort_values(
    by="Mean Onset Error"
).reset_index(drop=True)

# Save results
results_df.to_csv(OUTPUT_FILE, index=False)

# Display results
print("\nOPTIMIZED RHYTHM GRID ANALYSIS")
print("=" * 65)

print(f"Total Notes Analyzed: {len(onsets)}")

print("\nResults:")
print(results_df.to_string(index=False))

print("\nBest Candidate Grid:")
best = results_df.iloc[0]

print(f"Grid Duration: {best['Grid Duration']:.5f} seconds")
print(f"Best Phase Offset: {best['Best Phase Offset']:.5f} seconds")
print(f"Mean Onset Error: {best['Mean Onset Error']:.5f} seconds")
print(
    f"Alignment Within 30ms: "
    f"{best['Alignment Within 30ms (%)']:.2f}%"
)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nAnalysis completed.")