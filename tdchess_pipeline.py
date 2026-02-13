#!/usr/bin/env python3
# filepath: /Users/dsc/Codeberg/TDChess/tdchess_pipeline.py
"""
TDChess Training Pipeline - Implements TD(λ) learning in Python
"""

import os
import sys
import argparse
import subprocess
import json
import numpy as np
import torch
import time
from pathlib import Path

# Import train_neural.py functionality
from train_neural import train_model, ChessNet, ChessDataset

class TDChessTraining:
    """Manages the complete training pipeline for TDChess."""
    
    def __init__(self, model_dir="./model", iterations=250, games_per_iteration=250,
                 lambda_value=0.7, temperature=0.8, learning_rate=0.001):
        """Initialize the training pipeline."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.iterations = iterations
        self.games_per_iteration = games_per_iteration
        self.lambda_value = lambda_value
        self.temperature = temperature
        self.learning_rate = learning_rate
        
        # Make sure TDChess executable is available
        self.tdchess_exe = Path("./build/TDChess")
        if not self.tdchess_exe.exists():
            raise FileNotFoundError(f"TDChess executable not found at {self.tdchess_exe}")
            
        # Initial model and dataset paths - use .pt extension for PyTorch
        self.initial_model = self.model_dir / "chess_model_iter_0.pt"
        self.initial_dataset = self.model_dir / "initial_dataset.json"
        
        # Device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def ensure_initial_model(self):
        """Make sure we have an initial model, create one if needed."""
        if not self.initial_model.exists():
            print(f"No initial model found at {self.initial_model}")
            
            # Check for initial dataset, generate if needed
            if not self.initial_dataset.exists():
                print(f"Generating initial dataset with classical evaluation...")
                self.generate_initial_dataset()
            
            # Train initial model
            print(f"Training initial model...")
            train_model(
                str(self.initial_dataset),
                str(self.initial_model),
                epochs=250,
                batch_size=128,
                learning_rate=self.learning_rate
            )
    
    def generate_initial_dataset(self, num_positions=10000, max_depth=4):
        """Generate initial dataset using classical evaluation."""
        cmd = [
            str(self.tdchess_exe),
            "generate-dataset",
            str(self.initial_dataset),
            str(num_positions),
            str(max_depth)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    
    def generate_self_play_games(self, model_path, output_games_path):
        """Generate self-play games using the TDChess executable."""
        cmd = [
            str(self.tdchess_exe),
            "generate-self-play",  # New command to be added to TDChess
            str(model_path),
            str(output_games_path),
            str(self.games_per_iteration),
            str(self.temperature)
        ]
        
        print(f"Generating {self.games_per_iteration} self-play games...")
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return output_games_path
    
    def apply_td_lambda(self, games_path, output_dataset_path):
        """
        Apply TD(λ) learning algorithm to self-play games.
        This is the Python implementation of the TD(λ) algorithm.
        """
        print(f"Applying TD(λ) with λ={self.lambda_value} to games from {games_path}")
        
        # Load games data
        with open(games_path, 'r') as f:
            games_data = json.load(f)
        
        # Process each game to calculate TD targets
        positions = []
        td_targets = []
        
        for game in games_data['games']:
            # Extract game positions and result
            game_positions = game['positions']
            game_result = game['result']  # 1.0 (white win), 0.0 (draw), -1.0 (black win)
            
            # Apply TD(λ) to calculate targets for each position
            for i, position in enumerate(game_positions):
                # Skip last position (no future positions to learn from)
                if i == len(game_positions) - 1:
                    continue
                
                # Current evaluation (from current player's perspective)
                current_eval = position['evaluation']
                
                # Calculate TD(λ) target
                td_target = 0.0
                lambda_power = 1.0
                normalization = 0.0
                
                # Look ahead to future positions
                for k in range(1, len(game_positions) - i):
                    future_pos = game_positions[i + k]
                    future_eval = future_pos['evaluation']
                    
                    # Convert to white's perspective if necessary
                    if future_pos['side_to_move'] == 'BLACK':
                        future_eval = -future_eval
                    
                    # Add to TD target with λ weighting
                    td_target += lambda_power * future_eval
                    normalization += lambda_power
                    lambda_power *= self.lambda_value
                
                # Add final game result with remaining lambda weight
                # Game result is already from white's perspective
                td_target += lambda_power * game_result * 100.0  # Scale game result
                normalization += lambda_power
                
                # Normalize the target
                if normalization > 0:
                    td_target /= normalization
                
                # Clip target to reasonable range
                td_target = np.clip(td_target, -100.0, 100.0)
                
                # Add position and target to dataset
                positions.append(position['board'])
                td_targets.append(td_target)
        
        # Create dataset in the format expected by train_neural.py
        dataset = {
            'positions': [
                {
                    'board': pos,
                    'evaluation': target
                }
                for pos, target in zip(positions, td_targets)
            ]
        }
        
        # Save dataset
        with open(output_dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Created TD(λ) dataset with {len(positions)} positions at {output_dataset_path}")
        return output_dataset_path
    
    # Update output model paths
    def run_training_pipeline(self, start_iteration=1):
        """Run the complete training pipeline."""
        print(f"Starting TDChess training pipeline with {self.iterations} iterations")
        
        # Ensure we have an initial model
        self.ensure_initial_model()
        
        # Current model starts with the initial model
        current_model = self.initial_model
        
        # Iterative training
        for i in range(start_iteration, self.iterations + 1):
            print(f"\n--- Iteration {i} of {self.iterations} ---")
            
            # Paths for this iteration - use .pt extension
            games_path = self.model_dir / f"self_play_games_iter_{i}.json"
            td_dataset_path = self.model_dir / f"td_dataset_iter_{i}.json"
            output_model = self.model_dir / f"chess_model_iter_{i}.pt"
            
            # Step 1: Generate self-play games
            self.generate_self_play_games(current_model, games_path)
            
            # Step 2: Apply TD(λ) to the self-play games
            self.apply_td_lambda(games_path, td_dataset_path)
            
            # Step 3: Train new model using the TD(λ) dataset
            train_model(
                str(td_dataset_path),
                str(output_model),
                100,
                batch_size=128,
                learning_rate=self.learning_rate
            )
            
            # Update current model for next iteration
            current_model = output_model
            print(f"Completed iteration {i}. New model: {current_model}")
        
        print(f"Training pipeline complete! Final model: {current_model}")
        return current_model

def main():
    """Main entry point for TDChess training pipeline."""
    parser = argparse.ArgumentParser(description='TDChess Training Pipeline')
    parser.add_argument('--model-dir', type=str, default='./model', help='Directory for models and datasets')
    parser.add_argument('--iterations', type=int, default=50, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=250, help='Number of self-play games per iteration')
    parser.add_argument('--lambda', dest='lambda_value', type=float, default=0.7, help='TD(λ) parameter')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for move selection')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for neural network training')
    parser.add_argument('--start-iter', type=int, default=1, help='Starting iteration number')
    
    args = parser.parse_args()
    
    pipeline = TDChessTraining(
        model_dir=args.model_dir,
        iterations=args.iterations,
        games_per_iteration=args.games,
        lambda_value=args.lambda_value,
        temperature=args.temperature,
        learning_rate=args.learning_rate
    )
    
    pipeline.run_training_pipeline(start_iteration=args.start_iter)

if __name__ == '__main__':
    main()
