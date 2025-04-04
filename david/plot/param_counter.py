#!/usr/bin/env python3
import os
import sys
import glob
import re
import argparse
import numpy as np
from prettytable import PrettyTable
from safetensors.torch import load_file

def count_parameters(file_path):
    """Count weights and biases in a .safetensors file."""
    weight_count = 0
    bias_count = 0
    total_params = 0
    
    try:
        # Load the safetensors file
        tensors = load_file(file_path)
        
        for tensor_name, tensor in tensors.items():
            # Get tensor shape as a tuple
            tensor_shape = tensor.shape
            
            # Calculate number of parameters
            param_count = np.prod(tensor_shape)
            total_params += param_count
            
            # Count as weight or bias based on tensor name
            if "w_" in tensor_name.lower():
                weight_count += param_count
            elif "b_" in tensor_name.lower():
                bias_count += param_count
        
        return {
            "weights": weight_count,
            "biases": bias_count,
            "total": total_params
        }
    except Exception as e:
        raise Exception(f"Failed to parse file: {str(e)}")

def format_number(num):
    """Format large numbers with commas."""
    return f"{num:,}"

def analyze_folder(folder_path, file_pattern=None):
    """Analyze .safetensors files in a folder, with optional regex filtering."""
    folder_total = {"weights": 0, "biases": 0, "total": 0}
    file_count = 0
    
    # Find all .safetensors files in the folder
    safetensors_files = glob.glob(os.path.join(folder_path, "**", "*.safetensors"), recursive=True)
    
    # Filter out files with "model" in the name
    filtered_files = [f for f in safetensors_files if "model" not in os.path.basename(f).lower()]
    
    # Apply regex pattern if provided
    if file_pattern:
        try:
            pattern = re.compile(file_pattern)
            filtered_files = [f for f in filtered_files if pattern.search(os.path.basename(f))]
        except re.error:
            print(f"Warning: Invalid regex pattern '{file_pattern}'. Using all files.")
    
    if not filtered_files:
        return {"model_name": os.path.basename(folder_path), "weights": 0, "biases": 0, "total": 0, "file_count": 0}
    
    for file_path in filtered_files:
        try:
            file_stats = count_parameters(file_path)
            
            # Update folder totals
            folder_total["weights"] += file_stats["weights"]
            folder_total["biases"] += file_stats["biases"]
            folder_total["total"] += file_stats["total"]
            file_count += 1
        except Exception as e:
            # Just skip problematic files
            pass
    
    return {
        "model_name": os.path.basename(folder_path),
        "weights": folder_total["weights"],
        "biases": folder_total["biases"],
        "total": folder_total["total"],
        "file_count": file_count
    }

def find_subdirectories(base_dir):
    """Find all subdirectories in the given directory."""
    return [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]

def main():
    parser = argparse.ArgumentParser(description="Analyze .safetensors files in specified folders")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folders", "-f", nargs="+", help="Specific folders to analyze")
    group.add_argument("--directory", "-d", help="Global directory to recursively analyze all subdirectories")
    parser.add_argument("--pattern", "-p", help="Regex pattern to filter files by basename (e.g., 'sae.[0-9]')")
    parser.add_argument("--model-filter", "-m", help="Regex pattern to filter models by name (e.g., 'topk.*')")
    args = parser.parse_args()
    
    folders_to_analyze = []
    
    if args.directory:
        # Find all subdirectories in the global directory
        parent_dir = args.directory
        if not os.path.exists(parent_dir) or not os.path.isdir(parent_dir):
            print(f"Error: '{parent_dir}' is not a valid directory.")
            sys.exit(1)
            
        subdirs = find_subdirectories(parent_dir)
        if not subdirs:
            # If no subdirectories, analyze the parent directory itself
            folders_to_analyze = [parent_dir]
        else:
            folders_to_analyze = subdirs
            
        # Apply model name filter if specified
        if args.model_filter:
            try:
                model_pattern = re.compile(args.model_filter)
                folders_to_analyze = [f for f in folders_to_analyze if model_pattern.search(os.path.basename(f))]
                if not folders_to_analyze:
                    print(f"No models matched the pattern '{args.model_filter}'")
                    sys.exit(0)
            except re.error:
                print(f"Warning: Invalid model regex pattern '{args.model_filter}'. Using all folders.")
    else:
        folders_to_analyze = args.folders
    
    all_results = []
    for folder in folders_to_analyze:
        if os.path.exists(folder) and os.path.isdir(folder):
            result = analyze_folder(folder, args.pattern)
            all_results.append(result)
        else:
            print(f"Error: '{folder}' is not a valid directory.")
    
    # Only print models with parameters
    valid_results = [r for r in all_results if r["file_count"] > 0]
    
    # Create and print concise model table
    model_table = PrettyTable()
    model_table.field_names = ["Model", "Weight Parameters", "Bias Parameters", "Total Parameters"]
    
    # Sort by total parameters
    valid_results.sort(key=lambda x: x["total"], reverse=True)
    
    for result in valid_results:
        model_table.add_row([
            result["model_name"],
            format_number(result["weights"]),
            format_number(result["biases"]),
            format_number(result["total"])
        ])
    
    print("\nModel Parameter Summary")
    if args.pattern:
        print(f"File filter: '{args.pattern}'")
    if args.model_filter:
        print(f"Model filter: '{args.model_filter}'")
    print(model_table)

if __name__ == "__main__":
    main()
