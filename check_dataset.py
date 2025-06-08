#!/usr/bin/env python3
"""
Check the TDChess dataset for issues that might prevent convergence
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def analyze_dataset(dataset_path):
    """Analyze a chess dataset to check for issues"""
    print(f"Analyzing dataset: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if "positions" not in data:
        print("ERROR: Dataset missing 'positions' key")
        return
    
    positions = data["positions"]
    print(f"Dataset contains {len(positions)} positions")
    
    # Check evaluations
    evals = [pos["evaluation"] for pos in positions]
    eval_mean = np.mean(evals)
    eval_std = np.std(evals)
    eval_min = np.min(evals)
    eval_max = np.max(evals)
    
    print(f"Evaluation stats:")
    print(f"  Mean: {eval_mean:.2f}")
    print(f"  Std:  {eval_std:.2f}")
    print(f"  Min:  {eval_min:.2f}")
    print(f"  Max:  {eval_max:.2f}")
    
    # Check for missing or corrupted tensor data
    tensor_shapes = []
    for i, pos in enumerate(positions[:100]):  # Check first 100 positions
        if "board" not in pos:
            print(f"ERROR: Position {i} missing 'board' key")
            continue
        
        if "tensor" not in pos["board"]:
            print(f"ERROR: Position {i} missing 'tensor' data")
            continue
        
        tensor_shapes.append(len(pos["board"]["tensor"]))
    
    if tensor_shapes:
        if len(set(tensor_shapes)) > 1:
            print(f"WARNING: Inconsistent tensor shapes: {set(tensor_shapes)}")
        else:
            print(f"Tensor shape looks consistent: {tensor_shapes[0]} elements")
    
    # Plot evaluation distribution
    plt.figure(figsize=(10, 6))
    plt.hist(evals, bins=50)
    plt.title('Evaluation Distribution')
    plt.xlabel('Evaluation (pawns)')
    plt.ylabel('Count')
    plt.savefig(Path(dataset_path).with_suffix('.dist.png'))
    print(f"Saved distribution plot to {Path(dataset_path).with_suffix('.dist.png')}")
    
    # Check for constant evaluations
    unique_evals = set(evals)
    if len(unique_evals) < 10:
        print(f"WARNING: Only {len(unique_evals)} unique evaluation values found")
        print(f"Unique values: {unique_evals}")

def main():
    parser = argparse.ArgumentParser(description='Analyze TDChess dataset')
    parser.add_argument('dataset', help='Path to dataset JSON file')
    args = parser.parse_args()
    
    analyze_dataset(args.dataset)

if __name__ == '__main__':
    main()
