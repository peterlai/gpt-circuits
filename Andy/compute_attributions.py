from attributor import IntegratedGradientAttributor, ManualAblationAttributor
import os
import json
import sys
sys.path.append('/workspace/gpt-circuits')

import argparse
from pathlib import Path

from utils import sorted_indices_by_value

from config.sae.training import options
from config.sae.models import sae_options, SAEVariant
from data.dataloaders import TrainingDataLoader
import torch

from models.gpt import GPT
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Model config")
    parser.add_argument("--load_from", type=str, help="Model weights to load")
    parser.add_argument("--save_to", type=str, help="Path to save the attributions")
    parser.add_argument("--data_dir", type=str, help="Directory containing the data")

    parser.add_argument("--attribution_method", type=str, choices=["ig", "ma"], default = 'ig', help="Attribution method to use, either 'ig' or 'ma'")

    parser.add_argument("--save_name", type=str, default = '', help="Name of experiment to save")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading")
    parser.add_argument("--num_batches", type=int, default=32, help="Number of batches to process")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps in ig path (ig only)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon value for ma (ma only)")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose output")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    config_name = args.config
    config = sae_options[config_name]

    model = SparsifiedGPT(config)
    model_path = Path(args.load_from)
    model = model.load(model_path, device=config.device)
    model.to(config.device)  # Move model to the appropriate device

    data_dir = args.data_dir
    batch_size = args.batch_size



    dataloader = TrainingDataLoader(
        dir_path=data_dir,
        B= batch_size,
        T=model.config.block_size,
        process_rank=0,
        num_processes=1,
        split="val",
    )

    if args.attribution_method == "ig":
        attributor = IntegratedGradientAttributor(model, dataloader, nbatches = args.num_batches, verbose=args.verbose, steps=args.steps)
        attributions = attributor.layer_by_layer()
    elif args.attribution_method == "ma":
        attributor = ManualAblationAttributor(model, dataloader, nbatches = args.num_batches, verbose=args.verbose, epsilon=args.epsilon)
        attributions = attributor.layer_by_layer()

    attributions_listed = {}
    for key in attributions.keys():
        attributions_listed[key] = sorted_indices_by_value(attributions[key])


    output_filename = args.save_to
    name = args.save_name
    if name == '':
        name = config_name + '_' + args.attribution_method

    if output_filename.endswith(".json"):
        path = Path(output_filename)
    else:
        path = os.path.join(output_filename, "attributions.json")

    # Load existing data if the file exists
    if os.path.isfile(path):
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                raise ValueError("attributions.json exists but is not valid JSON")
    else:
        data = {}

    # Add or update the entry
    data[name] = attributions_listed

    # Save it back to disk
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    





    