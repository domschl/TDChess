#!/usr/bin/env python3
# filepath: /Users/dsc/Codeberg/TDChess/diagnose_dataset.py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def diagnose_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    positions = data["positions"]
    print(f"Dataset has {len(positions)} positions")
    
    # Check evaluations
    evals = [pos["evaluation"] for pos in positions]
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

if __name__ == "__main__":
    import sys
    diagnose_dataset(sys.argv[1] if len(sys.argv) > 1 else "../model/initial_dataset.json")
    