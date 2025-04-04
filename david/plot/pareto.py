# %%
import matplotlib.pyplot as plt
import numpy as np

# Define models, parameter counts, and losses
models = [
    "topk-x40.shakespeare_64x4",
    "topk-staircase-noshare.shakespeare_64x4",
    "topk-staircase-share-all.shakespeare_64x4",
    "topk-staircase-share-bdec-tied.shakespeare_64x4",
    "topk-staircase-detach.shakespeare_64x4",
    "topk_staircase_test",
    "topk-staircase-share.shakespeare_64x4",
    "top5.shakespeare_64x4",
    "top20.shakespeare_64x4",
    "topk.shakespeare_64x4"
]

parameter_counts = np.array([
    1651520, 991040, 340800, 335744, 335680, 335680, 335680, 330560, 330560, 330560
])

loss_values = np.array([
    0.5706, 0.5077, 0.4533, 0.4368, 0.3163, 0.3163, 0.3163, 0.2701, 0.2701, 0.2701
])

# Compute Pareto frontier
pareto_indices = np.argsort(parameter_counts)
pareto_counts = parameter_counts[pareto_indices]
pareto_losses = loss_values[pareto_indices]

# Identify Pareto-optimal points
pareto_frontier_x = []
pareto_frontier_y = []
current_best = float('inf')

for count, loss in zip(pareto_counts, pareto_losses):
    if loss < current_best:
        pareto_frontier_x.append(count)
        pareto_frontier_y.append(loss)
        current_best = loss

# Plot all models
plt.figure(figsize=(10, 6))
plt.scatter(parameter_counts, loss_values, color='blue', label="Models")

# Annotate points
for model, x, y in zip(models, parameter_counts, loss_values):
    plt.text(x, y, model, fontsize=9, ha='right', va='bottom')

# Plot Pareto frontier
plt.plot(pareto_frontier_x, pareto_frontier_y, linestyle='--', color='red', marker='o', label="Pareto Frontier")

# Labels and title
plt.xlabel("Total Parameters")
plt.ylabel("Loss")
plt.title("Loss vs. Parameter Count with Pareto Frontier")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
