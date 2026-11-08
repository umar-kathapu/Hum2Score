import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(r"D:\Musical-Score-Generation")

INPUT_FILE = (
    BASE_DIR / "outputs" / "rhythm_analysis"
    / "rhythm_analysis_0000.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "rhythm_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_FILE)

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(15, 9))

# Graph 1: Note duration
axes[0].bar(
    range(len(df)),
    df["Duration"],
    width=0.7
)

axes[0].set_title("Musical Note Duration")
axes[0].set_xlabel("Note Index")
axes[0].set_ylabel("Duration (Seconds)")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

# Graph 2: Inter-note gap
gap_data = df["Gap"].dropna()

axes[1].bar(
    range(1, len(df)),
    gap_data,
    width=0.7
)

axes[1].axhline(
    y=0,
    linestyle="--",
    linewidth=1
)

axes[1].set_title("Inter-Note Gap")
axes[1].set_xlabel("Note Index")
axes[1].set_ylabel("Gap (Seconds)")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

# Save graph
output_file = OUTPUT_DIR / "rhythm_visualization_0000.png"

plt.savefig(output_file, dpi=300)
plt.show()

print("\nRHYTHM VISUALIZATION COMPLETED!")
print("Graph saved at:", output_file)
