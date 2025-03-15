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

# Setup for toy-v0
# DIRNAME="toy-v0"
# TRAIN_TOKEN_IDS=(7010 300553 393485 512822 780099 872699)
# VAL_TOKEN_IDS=(15524 69324 75324 85424)

# Setup for toy-local
DIRNAME="toy-local"
TRAIN_TOKEN_IDS=()
VAL_TOKEN_IDS=(15 1039 2063 3087 4111 5135 6159 7183)

# Process training data
process_token_ids "${DIRNAME}" "train" "${TRAIN_TOKEN_IDS[@]}"

# Process validation data
process_token_ids "${DIRNAME}" "val" "${VAL_TOKEN_IDS[@]}"