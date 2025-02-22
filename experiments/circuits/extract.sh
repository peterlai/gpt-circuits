#!/bin/bash
# Extracts a circuit using a given (i) a token index, (ii) an dirname, and (iii) a resampling strategy.
# Example usage: `source experiments/circuits/extract.sh 51 toy-resample true`

# Set first positional argument as the shard token idx (default to 0)
SHARD_TOKEN_IDX=${1:-0}

# Set the second positional argument as the dirname (default to "mri")
DIRNAME=${2:-mri}

# Set ablation strategy
SHOULD_RESAMPLE=${3:-true}

# Set token idx to be the shard token idx % 128
TOKEN_IDX=$((SHARD_TOKEN_IDX % 128))

# Set the sequence idx to be token_idx // 128 * 128
SEQUENCE_IDX=$((SHARD_TOKEN_IDX / 128 * 128))

# Set circuit name
CIRCUIT_NAME="train.0.$SEQUENCE_IDX.$TOKEN_IDX"
echo "Extracting circuit: $CIRCUIT_NAME"

# Setup trap to kill all child processes on script exit
trap 'kill $(jobs -p) 2>/dev/null' EXIT INT

# Extract nodes in parallel
for layer_idx in {0..4}; do
    python -m experiments.circuits.nodes \
        --sequence_idx=$SEQUENCE_IDX \
        --token_idx=$TOKEN_IDX \
        --layer_idx=$layer_idx \
        $([ "$SHOULD_RESAMPLE" = "false" ] && echo "--no-resample") &
done

# Wait for all processes to finish
wait

# Extract edges in parallel
for layer_idx in {0..3}; do
    python -m experiments.circuits.edges \
        --circuit=$CIRCUIT_NAME \
        --upstream_layer=$layer_idx \
        $([ "$SHOULD_RESAMPLE" = "false" ] && echo "--no-resample") &
done

# Wait for all processes to finish
wait

# Export the circuit
python -m experiments.circuits.export --circuit=$CIRCUIT_NAME --dirname=$DIRNAME