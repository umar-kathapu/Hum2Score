import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(r"D:\Musical-Score-Generation")

INPUT_FILE = BASE_DIR / "outputs" / "note_segmentation" / "notes_0000.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "note_segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load segmented notes
df = pd.read_csv(INPUT_FILE)

# Create visualization
plt.figure(figsize=(16, 7))

for _, row in df.iterrows():

    start = row["Start Time"]
    end = row["End Time"]
    note = row["Note"]

    plt.hlines(
        y=note,
        xmin=start,
        xmax=end,
        linewidth=5
    )

    plt.text(
        (start + end) / 2,
        note,
        note,
        fontsize=8,
        ha="center",
        va="bottom"
    )

plt.xlabel("Time (Seconds)")
plt.ylabel("Musical Notes")
plt.title("Musical Note Segmentation - 0000.wav")

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Save graph
output_file = OUTPUT_DIR / "note_visualization_0000.png"

plt.savefig(output_file, dpi=300)
plt.show()

print("NOTE VISUALIZATION COMPLETED!")
print("Total Notes:", len(df))
print("Graph saved at:", output_file)