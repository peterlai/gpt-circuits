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

# Samples to use
VAL_TOKEN_IDS=(2 1026 2050)

# Cluster resampling
process_token_ids "comparisons-cluster" "val" "${VAL_TOKEN_IDS[@]}"

# Cluster resampling
process_token_ids "comparisons-cluster-nopos" "val" "${VAL_TOKEN_IDS[@]}"

# Zero ablation
process_token_ids "comparisons-zero" "val" "${VAL_TOKEN_IDS[@]}"

# Conventional resampling
process_token_ids "comparisons-classic" "val" "${VAL_TOKEN_IDS[@]}"