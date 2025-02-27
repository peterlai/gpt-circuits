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

# Extract nodes in parallel
for layer_idx in {0..4}; do
    python -m experiments.circuits.nodes \
        --sequence_idx=$SEQUENCE_IDX \
        --split=$SPLIT \
        --token_idx=$TOKEN_IDX \
        --layer_idx=$layer_idx &
done

# Wait for all processes to finish
wait

# Extract edges in parallel
for layer_idx in {0..3}; do
    python -m experiments.circuits.edges \
        --circuit=$CIRCUIT_NAME \
        --upstream_layer=$layer_idx &
done

# Wait for all processes to finish
wait

# Export circuits using a set of thresholds
python -m experiments.circuits.export --circuit=$CIRCUIT_NAME --dirname=$DIRNAME --threshold=0.15
python -m experiments.circuits.export --circuit=$CIRCUIT_NAME --dirname=$DIRNAME --threshold=0.20
python -m experiments.circuits.export --circuit=$CIRCUIT_NAME --dirname=$DIRNAME --threshold=0.25