#!/usr/bin/env python3
# filepath: /Users/dsc/Codeberg/TDChess/diagnose_dataset.py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

def check_dataset_integrity(dataset_path):
    """
    Checks the integrity of the dataset file.
    - File exists
    - JSON loads
    - Top-level 'positions' key exists and is a list
    - Each entry has 'board'->'tensor' (list of 896 floats) and 'evaluation' (float/int)
    """
    if not os.path.isfile(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' does not exist.")
        return False

    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return False

    if "positions" not in data or not isinstance(data["positions"], list):
        print("Error: Dataset JSON must have a top-level 'positions' key with a list value.")
        return False

    for idx, entry in enumerate(data["positions"]):
        if "board" not in entry or "tensor" not in entry["board"]:
            print(f"Error: Entry {idx} missing 'board' or 'tensor'.")
            return False
        tensor = entry["board"]["tensor"]
        if not isinstance(tensor, list) or len(tensor) != 896:
            print(f"Error: Entry {idx} tensor is not a list of length 896.")
            return False
        if not all(isinstance(x, (float, int)) for x in tensor):
            print(f"Error: Entry {idx} tensor contains non-numeric values.")
            return False
        if "evaluation" not in entry or not isinstance(entry["evaluation"], (float, int)):
            print(f"Error: Entry {idx} missing or invalid 'evaluation'.")
            return False

    print("Dataset integrity check passed.")
    return True

def diagnose_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    positions = data["positions"]
    print(f"Dataset has {len(positions)} positions")
    
    # Check evaluations
    evals = [pos["evaluation"] for pos in positions]
    evals = np.clip(evals, -10000, 10000)
    print(f"Eval stats: min={min(evals):.2f}, max={max(evals):.2f}, mean={np.mean(evals):.2f}, std={np.std(evals):.2f}")
    
    # Check for tensor problems
    tensor_check = positions[0]["board"]["tensor"]
    print(f"First tensor shape: {len(tensor_check)}")
    print(f"Tensor stats: min={min(tensor_check):.2f}, max={max(tensor_check):.2f}, mean={np.mean(tensor_check):.2f}")
    
    # Plot evaluation distribution
    plt.figure(figsize=(10,6))
    plt.hist(evals, bins=50)
    plt.title("Evaluation Distribution")
    plt.savefig(Path(dataset_path).with_suffix('.hist.png'))
    print(f"Histogram saved to {Path(dataset_path).with_suffix('.hist.png')}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_dataset.py <dataset_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]

    if not check_dataset_integrity(dataset_path):
        print("Dataset integrity check failed. Aborting analysis.")
        sys.exit(1)

    diagnose_dataset(dataset_path)

if __name__ == "__main__":
    main()
