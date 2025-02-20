#!/bin/bash
TIMEOUT=$(command -v timeout || command -v gtimeout)
if [ -z "$TIMEOUT" ]; then
  echo "Error: timeout command not found. Please install coreutils or gtimeout."
  exit 1
fi

# List of shard token IDs to process
SHARD_TOKEN_IDS=(7010 17396 196593 221099 229218 300553 301857 352614 382875 393485 512822 677317 780099 872699 899938 946999)

for SHARD_TOKEN_ID in "${SHARD_TOKEN_IDS[@]}"; do
  echo "Processing SHARD_TOKEN_ID=${SHARD_TOKEN_ID}"

  # Extract circuits using resampling ablation
  TIMEOUT 60m ./experiments/circuits/extract.sh $SHARD_TOKEN_ID toy-resampling true

  # Extract circuits using zero ablation
  TIMEOUT 60m ./experiments/circuits/extract.sh $SHARD_TOKEN_ID toy-zero false
done
