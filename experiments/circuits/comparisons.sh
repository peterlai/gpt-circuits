#!/bin/bash
TIMEOUT=$(command -v timeout || command -v gtimeout)
if [ -z "$TIMEOUT" ]; then
  echo "Error: timeout command not found. Please install coreutils or gtimeout."
  exit 1
fi

# Generate shard token IDs
VAL_TOKEN_IDS=()
for i in {0..99}; do
  SHARD_TOKEN_ID=$((i * 1028 + 2))

  $TIMEOUT 60m ./experiments/circuits/extract.sh "comparisons-cluster" "val" $SHARD_TOKEN_ID
  $TIMEOUT 60m ./experiments/circuits/extract.sh "comparisons-cluster-nopos" "val" $SHARD_TOKEN_ID
  $TIMEOUT 60m ./experiments/circuits/extract.sh "comparisons-zero" "val" $SHARD_TOKEN_ID
  $TIMEOUT 60m ./experiments/circuits/extract.sh "comparisons-classic" "val" $SHARD_TOKEN_ID
done