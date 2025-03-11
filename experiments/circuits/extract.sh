#!/bin/bash
# Extracts a set of circuits using a given (i) a dirname, (ii) a split and (iii) and a token index.
# Example usage: `source experiments/circuits/extract.sh toy-local train 51`

# Set the first positional argument as the dirname
DIRNAME=${1}

# Set the second positional argument as the split
SPLIT=${2}

# Set the third positional argument as the shard token idx
SHARD_TOKEN_IDX=${3}

# Set token idx to be the shard token idx % 128
TOKEN_IDX=$((SHARD_TOKEN_IDX % 128))

# Set the sequence idx to be token_idx // 128 * 128
SEQUENCE_IDX=$((SHARD_TOKEN_IDX / 128 * 128))

# Set circuit name
CIRCUIT_NAME="$SPLIT.0.$SEQUENCE_IDX.$TOKEN_IDX"
echo "Extracting '$CIRCUIT_NAME'"

# Setup trap to kill all child processes on script exit
trap 'kill $(jobs -p) 2>/dev/null' EXIT INT

# Extract nodes and edges
python -m experiments.circuits.circuit --split=$SPLIT --sequence_idx=$SEQUENCE_IDX --token_idx=$TOKEN_IDX --threshold=0.2 --num_samples=256

# Export circuit to visualizer
python -m experiments.circuits.export --dirname=$DIRNAME --circuit=$CIRCUIT_NAME