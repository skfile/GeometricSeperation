import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("mnist_cluster_ratio_results.csv")

# Filter for 1000 and 2000 sample sizes
df = df[df['sample_size'].isin([1000, 2000])]

# Set up NeurIPS-quality style
plt.rcParams.update({
    "font.size": 7,
    "font.family": "sans-serif",
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(2.5, 2))  # Small, publication-style

for sample_size, color in zip([1000, 2000], ['#1f77b4', '#ff7f0e']):
    sub = df[df['sample_size'] == sample_size]
    ax.plot(
        sub['k'],
        sub['percentage_above_threshold'],
        marker='o',
        label=f"MNIST {sample_size}",
        color=color,
        linewidth=1,
        markersize=3,
    )

ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("% Well-Separated Pairs")
ax.set_title("MNIST Cluster Ratio: 1000 & 2000 Samples")
ax.set_xlim(left=df['k'].min(), right=df['k'].max())
ax.set_ylim(bottom=0)
# No grid lines
ax.grid(False)

plt.tight_layout()
plt.savefig("mnist_1000_2000_ratio_percentage_plot.png", dpi=300)
plt.show()