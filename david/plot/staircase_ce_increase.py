# %%
# Re-import necessary libraries since execution state was reset
import matplotlib.pyplot as plt

# Re-define data
models = [
    "topk-staircase-detach (8,16,24,32,40)",
    "topk (8,8,8,8,8)",
    "topk-staircase-share (40,40,40,40,40)",
    "topk-staircase-share (8,16,24,32,40)",
    "topk-staircase-no-share (8,16,24,32,40)",
    "topk-wide (40,40,40,40,40)"
]
values = [0.5706, 0.5077, 0.4533, 0.4368, 0.3163, 0.2701]

# Plotting the bar chart with updated X-axis label
plt.figure(figsize=(10, 6))
plt.barh(models, values, color='skyblue')
plt.xlabel('End-to-end cross entropy increase')
plt.ylabel('Models')
plt.title('Top-K Model Performance')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()

# %%