#!/bin/bash
# filepath: xavier/experiments/sweep_edge_ablation.sh

# Default parameters
NUM_SAMPLES=2
NUM_PROMPTS=3
SEED=125

# Parameters to loop over
RUN_INDEX="jsae"
EDGE_SELECTIONS=("gradient")
UPSTREAM_LAYERS=(3)
EDGE_SET=(10 29136 58262 87388 116514 145640 174766 203892 233018 262143 262144)
SAE_VARIANTS=("1.0e-03" "1.2e-03" "1.5e-03" "1.8e-03" "2.2e-03" "2.7e-03" "3.3e-03" "3.9e-03" "4.7e-03" "5.6e-03" "6.8e-03" "1.0e-02")

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

# Data generation
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
        python xavier/experiments/magnitudes_logits_from_edges_mlp.py \
          --num-edges $NUM_EDGES \
          --upstream-layer-num $CURRENT_LAYER \
          --num-samples $NUM_SAMPLES \
          --num-prompts $NUM_PROMPTS \
          --edge-selection $EDGE_SELECTION \
          --sae-variant $SAE_VARIANT \
          --run-index $RUN_INDEX \
          --seed $SEED \
          2>&1 | tee "${LOG_DIR}/sae_${SAE_VARIANT}_layer${CURRENT_LAYER}_${EDGE_SELECTION}_edges_${NUM_EDGES}_${TIMESTAMP}.log"
        
        echo "Completed experiment with SAE variant $SAE_VARIANT, layer $CURRENT_LAYER, $NUM_EDGES edges, $EDGE_SELECTION strategy"
        echo ""

      done
    done
  done
done


echo "All data generation completed!"

# Basic plotting 
python xavier/experiments/basic_plotting.py \
  --run-index $RUN_INDEX \
  --sae-variant "${SAE_VARIANTS[@]}" \
  --edge-selections "${EDGE_SELECTIONS[@]}" \
  --upstream-layers "${UPSTREAM_LAYERS[@]}"

echo "Plotting completed!"