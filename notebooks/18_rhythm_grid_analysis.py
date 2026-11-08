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

OUTPUT_DIR = BASE_DIR / "outputs" / "rhythm_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD NOTE DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

df = df.sort_values("Start Time").reset_index(drop=True)

onsets = df["Start Time"].to_numpy()
durations = df["Duration"].to_numpy()

print("Total Notes:", len(df))

# ==========================================
# 3. CALCULATE ONSET INTERVALS
# ==========================================

onset_intervals = np.diff(onsets)

positive_intervals = onset_intervals[
    onset_intervals > 0
]

# ==========================================
# 4. TEST RHYTHMIC GRID CANDIDATES
# ==========================================

# Candidate grid sizes in seconds.
GRID_CANDIDATES = [
    0.0625,
    0.09375,
    0.125,
    0.1875,
    0.25,
    0.375,
    0.5
]

results = []

for grid in GRID_CANDIDATES:

    # Measure how closely onsets align to this grid.
    positions = onsets / grid

    nearest_positions = np.round(positions)

    errors = np.abs(
        positions - nearest_positions
    ) * grid

    mean_error = np.mean(errors)

    median_error = np.median(errors)

    within_tolerance = np.mean(
        errors <= grid * 0.10
    ) * 100

    results.append({

        "Grid Duration": grid,

        "Mean Onset Error": mean_error,

        "Median Onset Error": median_error,

        "Alignment Percentage": within_tolerance

    })

# ==========================================
# 5. CREATE RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Mean Onset Error"
).reset_index(drop=True)

# ==========================================
# 6. DURATION ANALYSIS
# ==========================================

duration_stats = {

    "Average Duration": np.mean(durations),

    "Median Duration": np.median(durations),

    "Shortest Duration": np.min(durations),

    "Longest Duration": np.max(durations)

}

# ==========================================
# 7. SAVE REPORTS
# ==========================================

results_df.to_csv(
    OUTPUT_DIR / "rhythm_grid_candidates.csv",
    index=False
)

# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n====================================")
print("RHYTHM GRID ANALYSIS COMPLETED")
print("====================================")

print("\nONSET INTERVAL STATISTICS")

print(
    "Average Onset Interval:",
    round(float(np.mean(positive_intervals)), 4)
)

print(
    "Median Onset Interval:",
    round(float(np.median(positive_intervals)), 4)
)

print("\nDURATION STATISTICS")

for name, value in duration_stats.items():

    print(f"{name}: {value:.4f} seconds")

print("\nRHYTHMIC GRID CANDIDATES")

print(results_df.to_string(index=False))

print("\nReport saved at:")
print(OUTPUT_DIR / "rhythm_grid_candidates.csv")