# %%
import matplotlib.pyplot as plt

# Re-define data
models = [
    "topk-staircase-detach (8,16,24,32,40)",
    "topk (8,8,8,8,8)",
    "topk-staircase-tied-all (40,40,40,40,40)",
    "topk-staircase-tied-chunks (8,16,24,32,40)",
    "topk-staircase-untied (8,16,24,32,40)",
    "topk-wide (40,40,40,40,40)"
]
values = [0.5706, 0.5077, 0.4533, 0.4368, 0.3163, 0.2701]
params = [335_680,  330_560, 340_800, 335_680, 991_040, 1_651_520]

# Define unique colors for each model
colors = ['red', 'magenta', 'green', 'cyan', 'orange', 'black']

# Plotting the bar chart
plt.figure(figsize=(10, 6))
bars = plt.barh(models, values, color='skyblue')
plt.xlabel('End-to-end cross entropy increase')
plt.ylabel('Models')
plt.title('Top-K Model Performance')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Annotate each bar with the numerical value
for bar, value in zip(bars, values):
    plt.text(
        bar.get_width() - 0.08,  # Position slightly to the right of the bar
        bar.get_y() + bar.get_height() / 2,  # Center vertically on the bar
        f'{value:.4f}',  # Format the value
        va='center',  # Align vertically
        ha='left',  # Align horizontally
        fontsize=12,  # Large font size
        color='black'  # Black text color
    )

plt.tight_layout()
plt.show()

# Plotting loss vs. parameter count
plt.figure(figsize=(8, 6))

# Scatter plot with unique colors and labeled legend
for model, x, y, color in zip(models, params, values, colors):
    plt.scatter(x, y, color=color, label=model, s=100)  # Larger size for visibility
    plt.annotate(f'({y:.4f}, {x})', (x, y), textcoords="offset points", xytext=(10,5), ha='center', fontsize=10)

plt.xlabel('Parameter Count')
plt.ylabel('End-to-end Cross Entropy Increase')
plt.title('Loss vs. Parameter Count')
plt.legend(title="Models", loc='upper right')  # Legend placed outside for clarity
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%