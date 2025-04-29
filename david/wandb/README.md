## Guide: Running Hyperparameter Sweeps with `sweep_dyt_alpha.yaml` using Weights & Biases

This guide explains how to use the provided `sweep_dyt_alpha.yaml` configuration file to perform hyperparameter optimization for your `training.gpt` Python script using Weights & Biases (W&B) Sweeps.

**Goal:** The objective of this sweep is to find the best combination of hyperparameters (`max_steps`, `lr_end`, `alpha_mlp`) that minimizes the `best_val_loss` metric for your GPT model training on the Shakespeare dataset.

### 1. Prerequisites

Before you begin, ensure you have the following:

1.  **Python Environment:** A working Python environment where your `training.gpt` script and its dependencies are installed.
2.  **Weights & Biases Account:** Sign up for a free account at [https://wandb.ai/site](https://wandb.ai/site).
3.  **W&B Library:** Install the W&B Python library:
    ```bash
    pip install wandb
    ```
4.  **W&B Login:** Log in to your W&B account from your terminal. You'll need your API key, which you can find at [https://wandb.ai/authorize](https://wandb.ai/authorize).
    ```bash
    wandb login
    ```
    Follow the prompts to enter your API key.

### 2. Understanding the `sweep_dyt_alpha.yaml` File

This YAML file defines the entire configuration for your hyperparameter sweep. Let's break down its components:

```yaml
# Command to execute for each run
command:
  - ${env} # Uses the current environment variables
  - python # The executable
  - -m
  - training.gpt # Your training script (run as a module)
  # W&B agent will automatically add swept parameters like --max_steps=VALUE --lr_end=VALUE etc.

# Sweep strategy
method: bayes # Uses Bayesian optimization to intelligently choose the next parameters to try

# Metric to optimize
metric:
  name: best_val_loss # The specific metric reported by your script that W&B should track
  goal: minimize # Aim to find parameters that result in the lowest value for this metric

# Parameters to explore
parameters:
  # --- Fixed parameters (applied to all runs) ---
  config:
    value: "shakespeare_64x4_dyt"
  wandb_project:
    value: "gpt-sweep-shakespeare_dyt" 
  norm_strategy: # Example of fixing a specific hyperparameter
    value: "DynamicTanh" # Ensures 'DynamicTanh' is used for norm_strategy in all runs

  # --- Hyperparameters to sweep ---
  max_steps:
    # Defines a list of discrete values to try for 'max_steps'
    values: [5000, 7500, 10000, 12500, 15000, 20000]
  lr_end: # Corresponds to 'min_lr' in your script's config (as per comment)
    # Samples values logarithmically between min and max (good for learning rates)
    distribution: log_uniform_values
    min: 5e-6 # Minimum value for lr_end
    max: 5e-4 # Maximum value for lr_end
  alpha_mlp: # Corresponds to 'gpt_config.alpha_mlp' in your script's config (as per comment)
    # Samples values logarithmically between min and max
    distribution: log_uniform_values
    min: 0.1  # Minimum value for alpha_mlp
    max: 10.0 # Maximum value for alpha_mlp

```

**Key Concepts:**

* **`command`**: Defines the base command W&B will execute for each trial (run) in the sweep.
    * `${env}` tells W&B to use the environment variables from where you run the `wandb agent`.
    * `python -m training.gpt` is how you run your script.
    * `--config=shakespeare_64x4_dyt` and `--wandb-project=gpt-sweep-shakespeare` are base arguments passed to your script.
    * **Important:** The W&B agent will *automatically append* the hyperparameters defined in the `parameters` section to this command for each run (e.g., `--max_steps=10000 --lr_end=1e-5 --alpha_mlp=0.5`). Your `training.gpt` script **must** be able to parse these command-line arguments (e.g., using `argparse`) and use them to override its default or config-file settings.
* **`method: bayes`**: Selects Bayesian Optimization as the search strategy. This method uses results from previous runs to make informed decisions about which hyperparameter combinations to try next, often finding good results faster than random search.
* **`metric`**: Tells W&B what value to track (`name: best_val_loss`) and whether to maximize or minimize it (`goal: minimize`). Your `training.gpt` script must log this exact metric name to W&B (e.g., using `wandb.log({'best_val_loss': value})`).
* **`parameters`**: This is the core of the sweep definition.
    * **Fixed Values:** Parameters like `wandb_project` or `norm_strategy` have a single `value` and will be the same for all runs initiated by this sweep configuration. They are passed as command-line arguments to your script.
    * **Swept Values:** Parameters like `max_steps`, `lr_end`, and `alpha_mlp` define the search space.
        * `values`: Provides a list of explicit values to try (`max_steps`).
        * `distribution`: Specifies how to sample values (e.g., `log_uniform_values` is suitable for parameters like learning rates where changes often have multiplicative effects). `min` and `max` define the range for the distribution.

### 3. Running the Sweep

Follow these steps in your terminal:

1.  **Navigate to Project Directory:** Change directory (`cd`) to the root directory where your `training.gpt` module and the `sweep_dyt_alpha.yaml` file are located.
2.  **Initialize the Sweep:** Run the following command:
    ```bash
    wandb sweep path/to/your/sweep_dyt_alpha.yaml
    ```
    (Replace `path/to/your/` if the yaml file isn't in the current directory).

    * **Output:** W&B will register the sweep configuration and print output similar to this:
        ```
        wandb: Creating sweep from: sweep_dyt_alpha.yaml
        wandb: Created sweep with ID: abc123xyz
        wandb: View sweep at: https://wandb.ai/<your_entity>/gpt-sweep-shakespeare/sweeps/abc123xyz
        wandb: Run sweep agent with: wandb agent <your_entity>/gpt-sweep-shakespeare/abc123xyz
        ```
    * **Take note** of the `Sweep ID` (e.g., `abc123xyz`) and the command provided to run the agent. The URL lets you monitor the sweep's progress in the W&B dashboard.

3.  **Start W&B Agents:** Agents are processes that pick up jobs (hyperparameter combinations) from the W&B sweep server, run your training script with those parameters, and report the results back.
    * Run the command provided in the previous step:
        ```bash
        wandb agent <your_entity>/gpt-sweep-shakespeare/<sweep_id>
        ```
        (Replace `<your_entity>` with your W&B username or team name and `<sweep_id>` with the actual ID from step 2).
    * The agent will start, ask the W&B server for a set of hyperparameters, run your `training.gpt` script with them, and repeat.
    * **Parallel Training:** You can run this `wandb agent` command on multiple machines, or multiple times on the same machine (if you have sufficient resources like GPUs), to execute sweep runs in parallel and speed up the optimization process. Each agent will independently fetch and run different hyperparameter combinations.

### 4. Monitoring the Sweep

* Open the URL provided when you initialized the sweep (e.g., `https://wandb.ai/<your_entity>/gpt-sweep-shakespeare/sweeps/abc123xyz`).
* The W&B dashboard provides powerful tools to visualize the sweep's progress:
    * See runs plotted based on their metrics (e.g., `best_val_loss` vs. `max_steps`).
    * Analyze parameter importance to see which hyperparameters affect the outcome most.
    * Compare runs directly.
    * Identify the best-performing runs and their corresponding hyperparameter settings.

### 5. Stopping the Sweep

* **Stop Agents:** You can stop individual agent processes by pressing `Ctrl+C` in the terminal where they are running.
* **Stop the Sweep Controller:** To prevent any new agents from picking up jobs, you can stop the sweep itself from the W&B UI (usually via a button on the sweep page). Existing runs will typically continue to completion unless the agents are stopped manually.

---

This detailed guide should help you effectively set up, run, and understand your hyperparameter sweep using W&B! Remember to ensure your `training.gpt` script correctly parses the command-line arguments provided by the W&B agent and logs the specified `metric`.