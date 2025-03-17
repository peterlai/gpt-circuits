#!/bin/bash
TIMEOUT=$(command -v timeout || command -v gtimeout)
if [ -z "$TIMEOUT" ]; then
  echo "Error: timeout command not found. Please install coreutils or gtimeout."
  exit 1
fi

# Config
MODEL_NAME="e2e.jumprelu.shakespeare_64x4"

trap "echo Exited!; exit;" SIGINT SIGTERM

# Generate shard token IDs
VAL_TOKEN_IDS=()
for i in {0..99}; do
  SEQUENCE_IDX=$((i * 1024))
  TOKEN_IDX=2
  CIRCUIT_NAME="val.0.$SEQUENCE_IDX.$TOKEN_IDX"

  for CONFIG_SUFFIX in "cluster-nopos" "cluster" "random-pos" "random" "zero"; do
    # Extract circuit
    echo "Extracting '$CIRCUIT_NAME' using '$CONFIG_SUFFIX'"
    python -m experiments.circuits.circuit \
      --split="val" --sequence_idx=$SEQUENCE_IDX --token_idx=$TOKEN_IDX --config_name="experiment-$CONFIG_SUFFIX" --skip_edges

    # Save results
    CIRCUIT_FILE="checkpoints/$MODEL_NAME/circuits/$CIRCUIT_NAME/config.json"
    DEST_DIR="checkpoints/$MODEL_NAME/ablation-comparisons/$CIRCUIT_NAME"

    if [ -f "$CIRCUIT_FILE" ]; then
      mkdir -p "$DEST_DIR"
      cp "$CIRCUIT_FILE" "$DEST_DIR/$CONFIG_SUFFIX.json"
    else
      echo "Warning: $CIRCUIT_FILE does not exist. Skipping."
    fi
  done
done