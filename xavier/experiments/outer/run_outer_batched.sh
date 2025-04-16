#!/bin/bash
# filepath: xavier/experiments/run_magnitudes_experiments.sh

# Default parameters
NUM_SAMPLES=2
NUM_PROMPTS=5
SEED=125

# Parameters to loop over
EDGE_SELECTIONS=("outer")
UPSTREAM_LAYERS=(3)
EDGE_SET=(10 20 30 40 50 60 70 80 90 100 110
        120 130 140 150 160 170 180 190 200)
SAE_VARIANTS=("jumprelu" "regularized" "top5" "top20" "topk")

# Create log directory if it doesn't exist
LOG_DIR="xavier/experiments/logs"
mkdir -p $LOG_DIR

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")

echo "Starting experiments..."
echo "- Number of samples: $NUM_SAMPLES"
echo "- Number of prompts: $NUM_PROMPTS"
echo "- Edge selection strategies: ${EDGE_SELECTIONS[*]}"
echo "- SAE variants: ${SAE_VARIANTS[*]}"
echo "- Random seed: $SEED"
echo ""

for SAE_VARIANT in "${SAE_VARIANTS[@]}"
do
  echo "Running experiments with SAE variant: $SAE_VARIANT"
  
  for EDGE_SELECTION in "${EDGE_SELECTIONS[@]}"
  do
    echo "Running experiments with edge selection strategy: $EDGE_SELECTION"
    
    for CURRENT_LAYER in "${UPSTREAM_LAYERS[@]}"
    do
      echo "Running experiments for upstream layer: $CURRENT_LAYER"
      
      # Run for logarithmically spaced number of edges
      for NUM_EDGES in "${EDGE_SET[@]}"
      do
        echo "Running experiment with SAE variant $SAE_VARIANT, layer $CURRENT_LAYER, $NUM_EDGES edges, $EDGE_SELECTION strategy..."
        
        # Run the Python script with the specified parameters
        python xavier/experiments/magnitudes_outer_batched.py \
          --num-edges $NUM_EDGES \
          --upstream-layer-num $CURRENT_LAYER \
          --num-samples $NUM_SAMPLES \
          --num-prompts $NUM_PROMPTS \
          --edge-selection $EDGE_SELECTION \
          --sae-variant $SAE_VARIANT \
          --seed $SEED \
          2>&1 | tee "${LOG_DIR}/sae_${SAE_VARIANT}_layer${CURRENT_LAYER}_${EDGE_SELECTION}_edges_${NUM_EDGES}_${TIMESTAMP}.log"
        
        echo "Completed experiment with SAE variant $SAE_VARIANT, layer $CURRENT_LAYER, $NUM_EDGES edges, $EDGE_SELECTION strategy"
        echo ""
        
        # Optional: add a small delay between runs
        sleep 1
      done
    done
  done
done

echo "All experiments completed!"
echo "Log files are available in: $LOG_DIR"