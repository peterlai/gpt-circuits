# GPT Circuits
Repository for training toy GPT-2 models and experimenting with sparse autoencoders.

## File Structure
* checkpoints - Training output
* [config](config) - Model and training configurations
* [data](data) - Datasets to be used for training
* [experiments](experiments) - Scripts for experiments
* exports - Exported JSON data
* [models](models) - GPT and SAE model definitions
* [training](training) - Model trainers

## Setup

### Environment
Python requirements are listed in [requirements.txt](requirements.txt). To install requirements, run:

```
pip install -r requirements.txt
```

### Datasets
Each dataset contains a [prepare.py](data/shakespeare/prepare.py) file. Run this file as a module from the root directory to prepare a dataset for use during training. Example:
```
python -m data.shakespeare.prepare
```

## Training

### GPT-2

Configurations are stored in [config/gpt](config/gpt). The trainer is located at [training/gpt.py](training/gpt.py). To run training, use:

```
python -m training.gpt --config=shakespeare_64x4
```

DDP is supported:

```
torchrun --standalone --nproc_per_node=8 -m training.gpt --config=shakespeare_64x4
```

### Sparse Autoencoders

Configurations are stored in [config/sae](config/sae). The Trainers are located at [training/sae](training/sae). To run training, use:

```
python -m training.sae.concurrent --config=standard.shakespeare_64x4 --load_from=shakespeare_64x4
```

### Cache Activations

For trainign multi-layer SAEs, it's faster to just shove the entire dataset through the model and cache the activations. This can be done with:

```
python -m spar.cache_activations --model=shakespeare_64x4 --dataset=shakespeare
```

We will still need to load the model and the validation dataset  and do forward passes for eval, but training will be a lot faster.