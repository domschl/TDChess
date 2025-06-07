#!/bin/bash
# filepath: /Users/dsc/Codeberg/TDChess/train_pipeline.sh

if [ ! -d "./model" ]; then
    mkdir model
fi

# Initial model
MODEL="./model/chess_model_iter_35.onnx"
INITIAL_DATASET="./model/initial_dataset.json"

# Number of iterations to run
ITERATIONS=50
START_ITERATION=36
TEMPERATURE=0.7

# Training parameters
GAMES_PER_ITERATION=250
LAMBDA=0.7
LEARNING_RATE=0.0001

echo "Starting TDChess training pipeline with $ITERATIONS iterations"

# Run initial dataset generation if no model exists
if [ ! -f "$MODEL" ]; then
    echo "No initial model found. Generating classical evaluation dataset..."
    if [ ! -f "$INITIAL_DATASET" ]; then
        echo "Initial dataset not found at $INITIAL_DATASET. Generating initial dataset..."
        ./build/TDChess generate-dataset "$INITIAL_DATASET" 10000 4
    fi
    echo "Training initial model..."
    python train_neural.py --dataset "$INITIAL_DATASET" --output "$MODEL" --epochs 200 --batch-size 128 --learning-rate $LEARNING_RATE
fi

# Iterative training
for ((i=$START_ITERATION; i<=$ITERATIONS; i++)); do
    echo "--- Iteration $i of $ITERATIONS ---"
    
    # Output model for this iteration
    OUTPUT_MODEL="./model/chess_model_iter_$i.onnx"

    # Run TD-Lambda training
    ./build/TDChess td-lambda "$MODEL" "$OUTPUT_MODEL" $GAMES_PER_ITERATION $LAMBDA $TEMPERATURE
    
    # Update current model for next iteration
    MODEL="$OUTPUT_MODEL"
    
    echo "Completed iteration $i. New model: $MODEL"
done

echo "Training pipeline complete! Final model: $MODEL"