#!/bin/bash
# filepath: xavier/experiments/run_magnitudes_experiments.sh

# Default parameters
UPSTREAM_LAYER=2
NUM_SAMPLES=2
NUM_PROMPTS=1
EDGE_SELECTION="gradient"
SEED=125

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --upstream-layer)
      UPSTREAM_LAYER="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --num-prompts)
      NUM_PROMPTS="$2"
      shift 2
      ;;
    --edge-selection)
      EDGE_SELECTION="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create log directory if it doesn't exist
LOG_DIR="xavier/experiments/logs"
mkdir -p $LOG_DIR

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")

echo "Starting experiments with the following parameters:"
echo "- Upstream layer: $UPSTREAM_LAYER"
echo "- Number of samples: $NUM_SAMPLES"
echo "- Number of prompts: $NUM_PROMPTS"
echo "- Edge selection strategy: $EDGE_SELECTION"
echo "- Random seed: $SEED"
echo ""

# Run for logarithmically spaced number of edges
# Starting with 10, then approximately doubling until 10000
# You can adjust these values based on your needs
for NUM_EDGES in 10 29136 58262 87388 116514 145640 174766 203892 233018 262144
do
  echo "Running experiment with $NUM_EDGES edges..."
  
  # Run the Python script with the specified parameters
  python xavier/experiments/magnitudes_logits_from_edges.py \
    --num-edges $NUM_EDGES \
    --upstream-layer-num $UPSTREAM_LAYER \
    --num-samples $NUM_SAMPLES \
    --num-prompts $NUM_PROMPTS \
    --edge-selection $EDGE_SELECTION \
    --seed $SEED \
    2>&1 | tee "${LOG_DIR}/edges_${NUM_EDGES}_${TIMESTAMP}.log"
  
  echo "Completed experiment with $NUM_EDGES edges"
  echo ""
  
  # Optional: add a small delay between runs
  sleep 1
done

echo "All experiments completed!"
echo "Log files are available in: $LOG_DIR"