# %%
"""
python -m david.test.safetensor checkpoints/topk-staircase-noshare.shakespeare_64x4
"""

import os
import argparse
from safetensors import safe_open

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Inspect .safetensors files in a specified folder.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing .safetensors files.")
    return parser.parse_args()

def inspect_safetensors_in_folder(folder_path: str):
    """
    Scans a folder for .safetensors files and prints the tensor keys and shapes for each file.
    
    :param folder_path: Path to the folder containing .safetensors files.
    """
    # List all .safetensors files in the folder and sort them
    safetensor_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".safetensors")])

    print(f"Inspecting {folder_path}")
    if not safetensor_files:
        print("No .safetensors files found in the folder.")
        return

    # Process each file in sorted order
    for file in safetensor_files:
        if file.startswith("model"):
            continue
        file_path = os.path.join(folder_path, file)
        print(f"\nInspecting: {file}")
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                print(f"  {key}: {f.get_tensor(key).shape}")

if __name__ == "__main__":
    args = parse_args()
    inspect_safetensors_in_folder(args.folder_path)
# %%