# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define L0 values as lower triangular matrices
l0_values_1 = [
    [9.9991],
    [7.3672, 2.6327],
    [2.3827, 3.2282, 4.3888],
    [1.2883, 2.1234, 2.9691, 3.6190],
    [1.1049, 1.1700, 1.5974, 1.8381, 4.2895]
]

l0_values_2 = [
    [9.9790],
    [2.1436, 7.8533],
    [0.0318, 0.0019, 9.9661],
    [0.0118, 0.0041, 0.0094, 9.9746],
    [0.0341, 0.0067, 0.0107, 0.3257, 9.6227]
]

# Function to create lower triangular matrix with NaNs
def create_heatmap_data(l0_values):
    matrix_size = len(l0_values)
    heatmap_data = np.full((matrix_size, matrix_size), np.nan)
    for i, row in enumerate(l0_values):
        heatmap_data[i, :len(row)] = row
    return heatmap_data

# Generate data
heatmap_data_1 = create_heatmap_data(l0_values_1)
heatmap_data_2 = create_heatmap_data(l0_values_2)

# Define color range
vmin, vmax = 0, 10

# Plot heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(heatmap_data_1, annot=True, fmt=".4f", cmap="coolwarm", linewidths=0.5, 
            mask=np.isnan(heatmap_data_1), ax=axes[0], vmin=vmin, vmax=vmax)
axes[0].set_xlabel("Index within chunk")
axes[0].set_ylabel("L0 Chunk Index")
axes[0].set_title("L0 per chunk, Staircase SAE")

sns.heatmap(heatmap_data_2, annot=True, fmt=".4f", cmap="coolwarm", linewidths=0.5, 
            mask=np.isnan(heatmap_data_2), ax=axes[1], vmin=vmin, vmax=vmax)
axes[1].set_xlabel("Index within chunk")
axes[1].set_ylabel("L0 Chunk Index")
axes[1].set_title("L0 per chunk, Staircase SAE (Detached Gradients)")

plt.tight_layout()
plt.show()

# %%
