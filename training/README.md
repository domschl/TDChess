# TDChess Training Scripts

This directory contains all Python scripts for dataset generation, neural network training, and the TD(λ) training pipeline for the TDChess engine.

- `generate_stockfish_dataset.py`: Generate datasets using Stockfish.
- `train_neural.py`: Train the neural network on chess positions.
- `tdchess_pipeline.py`: Full TD(λ) training pipeline.
- `diagnose_dataset.py`: Analyze and visualize dataset statistics.
- `check_dataset.py`: Check dataset integrity and convergence issues.

All scripts expect paths relative to the project root. The `model/` directory and `build/TDChess` executable are still referenced from the project root.
