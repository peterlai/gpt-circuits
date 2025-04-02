import json
import torch
import datetime
import dataclasses
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
from safetensors.torch import save_file, load_file

# Imports from the project
from config.sae.models import SAEConfig
from circuits import TokenlessEdge

@dataclass
class ExperimentParams:
    """Parameters of the experimental setup."""
    task: str
    ablator: str
    edges: frozenset[TokenlessEdge]
    edge_selection_strategy: str
    num_edges: int
    upstream_layer_idx: int
    num_samples: int
    num_prompts: int
    random_seed: int
    dataset_name: Optional[str] = None

@dataclass
class ExperimentResults:
    """Results of the experiment."""
    feature_magnitudes: torch.tensor
    logits: torch.tensor
    kl_divergence: torch.tensor
    execution_time: Optional[float] = None
    
@dataclass
class ExperimentOutput:
    """Complete standardized output for experiments."""
    experiment_id: str
    timestamp: datetime.datetime
    model_config: SAEConfig
    experiment_params: ExperimentParams
    results: ExperimentResults

    def to_safetensor(self, file_path: str) -> None:
        """
        Serialize the experiment output to SafeTensor format.
        
        This method saves:
        1. The raw tensor outputs in the results
        2. Metadata about the experiment as strings in the tensor's metadata
        
        Args:
            file_path: Path where to save the SafeTensor file
        """
        # Extract the tensor
        tensors = {"feature_magnitudes": self.results.feature_magnitudes, "logits": self.results.logits, "kl_divergence": self.results.kl_divergence}
        
        # Create metadata from non-tensor fields
        metadata = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp.isoformat(),
            "execution_time": str(self.results.execution_time) if self.results.execution_time is not None else "None",
            # Convert experiment parameters to JSON string
            "experiment_params": json.dumps(dataclasses.asdict(self.experiment_params)),
            # Convert model config to JSON string (excluding any non-serializable fields)
            "model_config": json.dumps({
                k: str(v) if k == 'device' else v
                for k, v in self.model_config.__dict__.items() 
                if not k.startswith('_') and k != 'gpt_config'
            })
        }

        save_file(tensors, file_path)
        # Save metadata separately
        with open(str(file_path) + ".metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)



    @classmethod
    def from_safetensor(cls, file_path: str) -> "ExperimentOutput":
        """
        Load an experiment output from a SafeTensor file.
        
        Args:
            file_path: Path to the SafeTensor file
            
        Returns:
            ExperimentOutput: Reconstructed experiment output object
        """
        # Load the safetensor file - handling different safetensors API versions

        metadata = {}
        with open(file_path + ".metadata.json", "r") as f:
            metadata = json.load(f)
            tensors = load_file(file_path)
        
        return tensors, metadata