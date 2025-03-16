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
VAL_TOKEN_IDS=(2 1026 2050 3074 4098 5122 6146 7170 8194 9218 10242 11266 12290 13314 14338 15362 16386 17410 18434 19458 20482 21486 22510 23534 24558 25582 26606 27630 28654 29678)

# Cluster resampling
process_token_ids "comparisons-cluster" "val" "${VAL_TOKEN_IDS[@]}"

# Cluster resampling
process_token_ids "comparisons-cluster-nopos" "val" "${VAL_TOKEN_IDS[@]}"

# Zero ablation
process_token_ids "comparisons-zero" "val" "${VAL_TOKEN_IDS[@]}"

# Conventional resampling
process_token_ids "comparisons-classic" "val" "${VAL_TOKEN_IDS[@]}"