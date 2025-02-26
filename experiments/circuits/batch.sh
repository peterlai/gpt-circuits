#!/bin/bash
TIMEOUT=$(command -v timeout || command -v gtimeout)
if [ -z "$TIMEOUT" ]; then
  echo "Error: timeout command not found. Please install coreutils or gtimeout."
  exit 1
fi

# Function to process a list of token IDs for a specific split
process_token_ids() {
  local model=$1
  local split=$2
  shift 2
  local token_ids=("$@")

  for token_id in "${token_ids[@]}"; do
    echo "Processing ${split}:${token_id}"
    $TIMEOUT 60m ./experiments/circuits/extract.sh $model $split $token_id
  done
}

# Training shard token IDs
TRAIN_TOKEN_IDS=(7010 300553 872699)

# Validation shard token IDs
VAL_TOKEN_IDS=(15524 85424)

# Process training data
process_token_ids "toy-v0" "train" "${TRAIN_TOKEN_IDS[@]}"

# Process validation data
process_token_ids "toy-v0" "val" "${VAL_TOKEN_IDS[@]}"