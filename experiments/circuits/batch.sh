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
  local threshold=$3
  shift 3
  local token_ids=("$@")

  for token_id in "${token_ids[@]}"; do
    echo "Processing ${split}:${token_id}"
    $TIMEOUT 180m ./experiments/circuits/extract.sh $model $split $token_id $threshold
  done
}

# Empty defaults
TRAIN_TOKEN_IDS=()
VAL_TOKEN_IDS=()

# DIRNAME="toy-v0"
# TRAIN_TOKEN_IDS=(7010 300553 393485 512822 780099 872699)
# VAL_TOKEN_IDS=(15 1039 15524 69324 75324 85424)

# DIRNAME="comparisons-random"
# DIRNAME="comparisons-zero"
# DIRNAME="comparisons-cluster"
# VAL_TOKEN_IDS=(1282 7554 8834 9218) # 3 char sequences
# VAL_TOKEN_IDS=(6159) # 16 char sequence

# DIRNAME="toy-local"
# TRAIN_TOKEN_IDS=()
# VAL_TOKEN_IDS=(15 1039 2063 3087 4111 5135 6159 7183)

# DIRNAME="toy-tiny-stories"
# VAL_TOKEN_IDS=(200491 201519 202547 203575 204603 205631 206659 207687 208715 209743 210771 211799)

# Process training data
process_token_ids "${DIRNAME}" "train" 0.15 "${TRAIN_TOKEN_IDS[@]}"

# Process validation data
process_token_ids "${DIRNAME}" "val" 0.25 "${VAL_TOKEN_IDS[@]}"