# %%
import argparse
import glob
import re
import sys
from pathlib import Path
import math

import matplotlib.pyplot as plt


def extract_info_from_debug_log(debug_log_path: Path):
    """
    Extracts the first sparsity coefficient (from the config in the first line)
    and max_steps (from the config in the last line) from a debug.log file.

    Args:
        debug_log_path: Path object pointing to the debug.log file.

    Returns:
        A tuple (sparsity_coeff, max_steps) if successful, otherwise (None, None).
        Values can be partially None if only one is found.
    """
    sparsity_coeff = None
    max_steps = None
    lines = []
    try:
        with open(debug_log_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                print(f"Warning: Debug log file is empty: {debug_log_path}", file=sys.stderr)
                return None, None

            # --- Extract Sparsity Coeff from First Line ---
            first_line = lines[0]
            # Regex to find the sparsity tuple within loss_coefficients
            # Updated regex to handle potential list format as well
            sparsity_match = re.search(r"loss_coefficients.*?\'sparsity\':\s*[\(\[]([\d\.\s,eE-]+)[\)\]]", first_line)
            if sparsity_match:
                try:
                    # Extract the first element of the sparsity tuple/list
                    sparsity_str = sparsity_match.group(1).split(',')[0].strip()
                    sparsity_coeff = float(sparsity_str)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse sparsity coefficient from first line in {debug_log_path}. Line: '{first_line.strip()}'. Error: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Unexpected error parsing sparsity coefficient from first line in {debug_log_path}. Line: '{first_line.strip()}'. Error: {e}", file=sys.stderr)
            else:
                 print(f"Warning: Could not find sparsity coefficient pattern in first line of {debug_log_path}. Line: '{first_line.strip()}'", file=sys.stderr)

            # --- Extract Max Steps from Last Line ---
            # Ensure there is a last line to read
            if len(lines) > 0:
                last_line = lines[-1]
                # Regex to find 'max_steps': number
                max_steps_match = re.search(r"'max_steps':\s*(\d+)", last_line)
                if max_steps_match:
                    try:
                        max_steps = int(max_steps_match.group(1))
                    except ValueError as e:
                        print(f"Warning: Could not parse max_steps from last line in {debug_log_path}. Line: '{last_line.strip()}'. Error: {e}", file=sys.stderr)
                    except Exception as e:
                        print(f"Warning: Unexpected error parsing max_steps from last line in {debug_log_path}. Line: '{last_line.strip()}'. Error: {e}", file=sys.stderr)
                else:
                    # Only warn if the line looks like it *should* contain config (e.g., starts with '{')
                    # Or maybe just always warn if not found? Let's warn.
                    print(f"Warning: Could not find 'max_steps' pattern in last line of {debug_log_path}. Line: '{last_line.strip()}'", file=sys.stderr)
            else:
                 # This case is already handled by the initial check for empty lines, but included for clarity
                 print(f"Warning: Cannot read last line for max_steps as file is empty: {debug_log_path}", file=sys.stderr)


        return sparsity_coeff, max_steps

    except FileNotFoundError:
        # Handled in main loop
        return None, None
    except Exception as e:
        print(f"Error reading or processing {debug_log_path}: {e}", file=sys.stderr)
        return None, None


def extract_metrics_from_eval_log(eval_log_path: Path):
    """
    Extracts the compound CE loss increase and the sum of ∇_l1 values
    from the last relevant line in an eval.log file.

    Args:
        eval_log_path: Path object pointing to the eval.log file.

    Returns:
        A tuple (loss_increase, nabla_l1_sum) if successful, otherwise (None, None).
    """
    loss_increase = None
    nabla_l1_sum = None
    lines = []
    try:
        with open(eval_log_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                print(f"Warning: Eval log file is empty: {eval_log_path}", file=sys.stderr)
                return None, None

            # Search backwards for the last line containing both metrics
            last_eval_line = None
            for line in reversed(lines):
                # Updated to look for the specific metrics more robustly
                if "compound_ce_loss_increase" in line and "∇_l1" in line:
                    last_eval_line = line
                    break

            if not last_eval_line:
                print(f"Warning: Could not find a line containing both 'compound_ce_loss_increase' and '∇_l1' in {eval_log_path}", file=sys.stderr)
                return None, None

            # --- Extract compound_ce_loss_increase ---
            # Use a more specific regex to avoid capturing other numbers
            loss_increase_match = re.search(r"compound_ce_loss_increase\s+([\-\+]?[\d\.eE]+)", last_eval_line)
            if loss_increase_match:
                try:
                    loss_increase = float(loss_increase_match.group(1))
                except ValueError as e:
                     print(f"Warning: Could not parse compound_ce_loss_increase from line in {eval_log_path}. Line: '{last_eval_line.strip()}'. Error: {e}", file=sys.stderr)
                except Exception as e:
                     print(f"Warning: Unexpected error parsing loss increase from line in {eval_log_path}. Line: '{last_eval_line.strip()}'. Error: {e}", file=sys.stderr)
            else:
                 print(f"Warning: Could not find 'compound_ce_loss_increase' pattern in relevant line of {eval_log_path}. Line: '{last_eval_line.strip()}'", file=sys.stderr)


            # --- Extract and sum ∇_l1 values ---
            # Use a more specific regex to capture the list of numbers after ∇_l1
            nabla_l1_match = re.search(r"∇_l1\s+((?:[\-\+]?[\d\.eE]+\s*)+)", last_eval_line)
            if nabla_l1_match:
                try:
                    nabla_l1_values_str = nabla_l1_match.group(1).strip()
                    # Handle potential multiple spaces between numbers
                    nabla_l1_values = [float(v) for v in re.split(r'\s+', nabla_l1_values_str) if v]
                    if nabla_l1_values: # Ensure list is not empty after split
                        nabla_l1_sum = sum(nabla_l1_values)
                    else:
                        print(f"Warning: Parsed empty list for ∇_l1 values in {eval_log_path}. Line: '{last_eval_line.strip()}'", file=sys.stderr)

                except ValueError as e:
                    print(f"Warning: Could not parse ∇_l1 values from line in {eval_log_path}. Line: '{last_eval_line.strip()}'. Error: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Unexpected error parsing ∇_l1 values from line in {eval_log_path}. Line: '{last_eval_line.strip()}'. Error: {e}", file=sys.stderr)
            else:
                print(f"Warning: Could not find '∇_l1' pattern in relevant line of {eval_log_path}. Line: '{last_eval_line.strip()}'", file=sys.stderr)


        # Return None for a value if it couldn't be parsed
        return loss_increase, nabla_l1_sum

    except FileNotFoundError:
         # Don't print a warning here, handled in main loop
        return None, None
    except Exception as e:
        print(f"Error reading or processing {eval_log_path}: {e}", file=sys.stderr)
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Extract summed ∇_l1 and loss increase from eval.log, label with sparsity coeff from debug.log, "
                    "extract max_steps from debug.log, and plot them, separating runs with max_steps=20000."
    )
    parser.add_argument(
        "path_pattern",
        help="Glob pattern for checkpoint directories (e.g., 'checkpoints/jsae*'). Quote the pattern if it contains wildcards."
    )
    args = parser.parse_args()

    data_points = [] # Will store tuples: (nabla_l1_sum, loss_increase, sparsity_coeff, max_steps)
    potential_paths = glob.glob(args.path_pattern)
    directories = [Path(p) for p in potential_paths if Path(p).is_dir()]

    if not directories:
        print(f"Error: No directories found matching pattern '{args.path_pattern}'", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(directories)} matching directories.")

    for dir_path in sorted(directories):
        debug_log_file = dir_path / "debug.log"
        eval_log_file = dir_path / "eval.log"
        print(f"Processing directory: {dir_path}")

        sparsity_coeff = None
        max_steps = None
        loss_increase = None
        nabla_l1_sum = None

        # Check for debug.log and extract info
        if debug_log_file.is_file():
            sparsity_coeff, max_steps = extract_info_from_debug_log(debug_log_file) # Updated call
            if sparsity_coeff is None:
                 print(f"  Warning: Could not get sparsity coefficient label from {debug_log_file}.", file=sys.stderr)
            if max_steps is None:
                 print(f"  Warning: Could not get max_steps from {debug_log_file}.", file=sys.stderr)
        else:
            print(f"  Warning: debug.log not found in {dir_path}, cannot get sparsity coefficient or max_steps.", file=sys.stderr)

        # Check for eval.log and extract metrics
        if eval_log_file.is_file():
            loss_increase, nabla_l1_sum = extract_metrics_from_eval_log(eval_log_file)
        else:
            print(f"  Warning: eval.log not found in {dir_path}", file=sys.stderr)

        # Only add data point if ALL required values were successfully extracted
        # Note: We now require max_steps as well for categorization
        if sparsity_coeff is not None and loss_increase is not None and nabla_l1_sum is not None and max_steps is not None:
             if nabla_l1_sum <= 0:
                 print(f"  Warning: Skipping directory {dir_path} because summed ∇_l1 is non-positive ({nabla_l1_sum}), cannot plot on log scale.", file=sys.stderr)
             else:
                print(f"  Successfully extracted data: Sparsity Coeff={sparsity_coeff}, Loss Increase={loss_increase}, Sum(∇_l1)={nabla_l1_sum}, Max Steps={max_steps}")
                # Store: (x_value, y_value, label_value, category_value)
                data_points.append((nabla_l1_sum, loss_increase, sparsity_coeff, max_steps))
        else:
            # Be more specific about what failed
            missing = []
            if sparsity_coeff is None: missing.append("sparsity_coeff")
            if loss_increase is None: missing.append("loss_increase")
            if nabla_l1_sum is None: missing.append("nabla_l1_sum")
            if max_steps is None: missing.append("max_steps")
            if missing: # Only print if something was actually missing (avoid printing if just nabla_l1_sum <= 0)
                print(f"  Failed to extract complete data quartet for {dir_path}. Missing: {', '.join(missing)}")
            elif not (nabla_l1_sum is not None and nabla_l1_sum > 0): # Handle the non-positive nabla case if it wasn't printed above
                 pass # Already handled by the specific warning above
            else: # Should not happen given the logic, but as a fallback
                 print(f"  Failed to extract complete data quartet for {dir_path} for unknown reason.")


    if not data_points:
        print("\nError: No valid data points (quartets of summed ∇_l1, loss increase, sparsity coeff, and max_steps) could be extracted.", file=sys.stderr)
        sys.exit(1)

    # Sort data points by summed ∇_l1 (x-axis value)
    data_points.sort(key=lambda x: x[0])

    # Separate data based on max_steps
    data_points_20k = [dp for dp in data_points if dp[3] == 20000]
    data_points_other = [dp for dp in data_points if dp[3] != 20000]

    # --- Create the plot ---
    plt.figure(figsize=(12, 7))

    # Plot data for max_steps != 20000 (if any)
    if data_points_other:
        nabla_l1_sums_other, loss_increases_other, sparsity_coeffs_other, _ = zip(*data_points_other)
        plt.scatter(nabla_l1_sums_other, loss_increases_other, label='Max Steps != 20k', zorder=5, color='blue', marker='o')
        plt.plot(nabla_l1_sums_other, loss_increases_other, linestyle='-', label='_nolegend_', zorder=4, color='blue', marker='o') # Use _nolegend_ to avoid duplicate legend entry for line
        # Label points
        for i, label_val in enumerate(sparsity_coeffs_other):
            plt.text(nabla_l1_sums_other[i]*1.05, loss_increases_other[i] * 1.01, f'{label_val:.1e}', fontsize=9, rotation=0, ha='left', va='bottom', color='blue')
        print(f"\nPlotting {len(data_points_other)} points with Max Steps != 20k (blue)")
    else:
        print("\nNo data points found with Max Steps != 20k.")


    # Plot data for max_steps == 20000 (if any)
    if data_points_20k:
        nabla_l1_sums_20k, loss_increases_20k, sparsity_coeffs_20k, _ = zip(*data_points_20k)
        plt.scatter(nabla_l1_sums_20k, loss_increases_20k, label='Max Steps == 20k', zorder=6, color='red', marker='^') # Use different marker
        plt.plot(nabla_l1_sums_20k, loss_increases_20k, linestyle='-', label='_nolegend_', zorder=5, color='red', marker='^') # Use _nolegend_
        # Label points
        for i, label_val in enumerate(sparsity_coeffs_20k):
            plt.text(nabla_l1_sums_20k[i]*1.05, loss_increases_20k[i] * 1.01, f'{label_val:.1e}', fontsize=9, rotation=0, ha='left', va='bottom', color='red')
        print(f"Plotting {len(data_points_20k)} points with Max Steps == 20k (red)")
    else:
        print("No data points found with Max Steps == 20k.")


    # Add the horizontal baselines
    baseline_loss = 0.4468
    plt.axhline(y=baseline_loss, color='grey', linestyle='--', label=f'Baseline Loss Increase ({baseline_loss:.4f})', zorder=3) # Changed color for better contrast

    # Add the horizontal line for random guessing loss ln(128)
    # Assuming vocab_size is 128 based on context file checkpoints/jsae.shakespeare_64x4-sparsity-0.0e+00/model.json
    vocab_size = 128
    random_guessing_loss = math.log(vocab_size)
    plt.axhline(y=random_guessing_loss, color='purple', linestyle=':', label=f'Random Guessing Loss (ln({vocab_size}) ≈ {random_guessing_loss:.4f})', zorder=3) # Changed color

    plt.xlabel("Sum(∇_l1) (from eval.log) - Log Scale")
    plt.ylabel("Compound CE Loss Increase (from eval.log)")
    plt.title("Sum(∇_l1) vs. Loss Increase (Labeled by Sparsity Coeff, Colored by Max Steps)") # Updated title

    plt.xscale('log') # Set x-axis to log scale

    plt.legend() # Display legend including the baselines and the two series
    plt.grid(True, which="both", linestyle='--', alpha=0.6) # Grid for both major and minor ticks on log scale
    plt.tight_layout() # Adjust plot to prevent labels overlapping
    print("\nPlotting results...")
    # Construct filename based on the pattern, replacing wildcards
    base_filename = args.path_pattern.replace('*', '_star_').replace('?', '_qmark_')
    # Sanitize further if needed (remove potentially problematic characters for filenames)
    base_filename = re.sub(r'[\\/:\s]', '_', base_filename)
    base_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '', base_filename) # Keep only safe chars
    save_path = f"{base_filename}_nabla_vs_loss_by_steps.png" # Updated filename
    print(f"Saving plot to: {save_path}")
    try:
        plt.savefig(save_path)
        print("Plot saved successfully.")
    except Exception as e:
        print(f"Error saving plot to {save_path}: {e}", file=sys.stderr)

    plt.show() # Display the plot

if __name__ == "__main__":
    main()
# %%