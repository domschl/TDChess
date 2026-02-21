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
import re
import concurrent.futures
import tempfile
from pathlib import Path

# Import train_neural.py functionality
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))
from train_neural import train_model, ChessNet, ChessDataset


class TDChessTraining:
    """Manages the complete training pipeline for TDChess."""

    def __init__(
        self,
        model_dir="../model",
        iterations=50,
        games_per_iteration=1000,
        lambda_value=0.7,
        temperature=0.8,
        learning_rate=0.001,
        num_workers=1,
        initial_positions=10000,
        initial_depth=4,
    ):
        """Initialize the training pipeline."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.iterations = iterations
        self.games_per_iteration = games_per_iteration
        self.lambda_value = lambda_value
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.initial_positions = initial_positions
        self.initial_depth = initial_depth

        # Make sure TDChess executable is available
        self.tdchess_exe = SCRIPT_DIR.parent / "build" / "TDChess"
        if not self.tdchess_exe.exists():
            raise FileNotFoundError(
                f"TDChess executable not found at {self.tdchess_exe}"
            )

        # Initial model and dataset paths - use .pt extension for PyTorch
        self.initial_model = self.model_dir / "chess_model_iter_0.pt"
        self.initial_dataset = self.model_dir / "initial_dataset.json"

        # Device for PyTorch
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using device: {self.device}")

    def find_latest_iteration(self):
        """Scan the model directory for the latest iteration model."""
        latest_iter = -1
        for f in self.model_dir.glob("chess_model_iter_*.pt"):
            try:
                # Extract number from chess_model_iter_N.pt
                # Use regex to be more robust
                match = re.search(r"chess_model_iter_(\d+)\.pt$", f.name)
                if match:
                    iter_num = int(match.group(1))
                    if iter_num > latest_iter:
                        latest_iter = iter_num
            except (ValueError, IndexError):
                continue
        return latest_iter

    def ensure_initial_model(self):
        """Make sure we have an initial model, create one if needed."""
        if not self.initial_model.exists():
            print(f"No initial model found at {self.initial_model}")

            # Check for initial dataset, generate if needed
            if not self.initial_dataset.exists():
                print(f"Generating initial dataset with classical evaluation...")
                self.generate_initial_dataset(
                    num_positions=self.initial_positions, max_depth=self.initial_depth
                )

            # Train initial model
            print(f"Training initial model...")
            train_model(
                str(self.initial_dataset),
                str(self.initial_model),
                epochs=1024,
                batch_size=128,
                learning_rate=self.learning_rate,
            )

    def generate_initial_dataset(self, num_positions=10000, max_depth=4):
        """Generate initial dataset using Stockfish and the Python script."""
        # Use the Python script for higher quality Stockfish evaluations
        script_path = SCRIPT_DIR / "generate_stockfish_dataset.py"
        cmd = [sys.executable, str(script_path), str(num_positions)]

        print(f"Running Stockfish dataset generation: {' '.join(cmd)}")
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(f"[StockfishGen] {line.strip()}", flush=True)
            process.stdout.close()

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

    def generate_self_play_games(self, model_path, output_games_path):
        """Generate self-play games using the TDChess executable."""
        if self.num_workers <= 1:
            base_seed = int(time.time())
            cmd = [
                str(self.tdchess_exe),
                "generate-self-play",
                str(model_path),
                str(output_games_path),
                str(self.games_per_iteration),
                str(self.temperature),
                str(base_seed),
            ]

            print(
                f"Generating {self.games_per_iteration} self-play games (seed {base_seed})..."
            )
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            return output_games_path

        # Parallel generation
        print(
            f"Generating {self.games_per_iteration} self-play games using {self.num_workers} workers..."
        )
        games_per_worker = self.games_per_iteration // self.num_workers
        remainder = self.games_per_iteration % self.num_workers

        # Use nano-second precision for the base seed to avoid collisions
        base_seed = int(time.time_ns() % (2**31 - 10000))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            futures = []

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                for i in range(self.num_workers):
                    # Add remainder games to the first worker
                    count = games_per_worker + (remainder if i == 0 else 0)
                    if count <= 0:
                        continue

                    chunk_path = tmpdir_path / f"games_chunk_{i}.json"
                    # Ensure each worker gets a very different seed
                    worker_seed = base_seed + i * 1000
                    futures.append(
                        executor.submit(
                            self._run_generation_chunk,
                            model_path,
                            chunk_path,
                            count,
                            worker_seed,
                        )
                    )

                # Wait for all chunks to finish
                for future in concurrent.futures.as_completed(futures):
                    future.result()

            # Merge chunks
            self._merge_game_files(tmpdir_path, output_games_path)

        return output_games_path

    def _run_generation_chunk(self, model_path, chunk_path, count, seed):
        """Run a single generation chunk process."""
        cmd = [
            str(self.tdchess_exe),
            "generate-self-play",
            str(model_path),
            str(chunk_path),
            str(count),
            str(self.temperature),
            str(seed),
        ]
        subprocess.run(cmd, check=True)

    def _merge_game_files(self, chunks_dir, output_path):
        """Merge multiple game JSON files into one."""
        merged_games = {"games": []}

        for chunk_file in chunks_dir.glob("games_chunk_*.json"):
            with open(chunk_file, "r") as f:
                data = json.load(f)
                if "games" in data:
                    merged_games["games"].extend(data["games"])

        with open(output_path, "w") as f:
            json.dump(merged_games, f, indent=2)
        print(f"Merged {len(merged_games['games'])} games into {output_path}")

    def apply_td_lambda(self, games_path, output_dataset_path):
        """
        Apply TD(λ) learning algorithm to self-play games.
        This is the Python implementation of the TD(λ) algorithm.
        """
        print(f"Applying TD(λ) with λ={self.lambda_value} to games from {games_path}")

        # Load games data
        with open(games_path, "r") as f:
            games_data = json.load(f)

        # Process each game to calculate TD targets
        positions = []
        td_targets = []

        for game in games_data["games"]:
            # Extract game positions and result
            game_positions = game["positions"]
            game_result = game[
                "result"
            ]  # 1.0 (white win), 0.0 (draw), -1.0 (black win)

            # Apply TD(λ) to calculate targets for each position
            for i, position in enumerate(game_positions):
                # Skip last position (no future positions to learn from)
                if i == len(game_positions) - 1:
                    continue

                # Current evaluation (from current player's perspective)
                current_eval = position["evaluation"]

                # Calculate TD(λ) target
                td_target = 0.0
                lambda_power = 1.0
                normalization = 0.0

                # Look ahead to future positions
                for k in range(1, len(game_positions) - i):
                    future_pos = game_positions[i + k]
                    future_eval = future_pos["evaluation"] * 100.0  # Convert pawn units back to centipawns

                    # Convert to white's perspective if necessary
                    if future_pos["side_to_move"] == "BLACK":
                        future_eval = -future_eval

                    # Add to TD target with λ weighting
                    td_target += lambda_power * future_eval
                    normalization += lambda_power
                    lambda_power *= self.lambda_value

                # Add final game result with remaining lambda weight
                # Game result is already from white's perspective (1.0, 0.0, -1.0)
                # Scale it to match the centipawn evaluation range (e.g. 2000.0 = 20 pawns)
                td_target += lambda_power * game_result * 2000.0  # Increased from 100.0
                normalization += lambda_power

                # Normalize the target
                if normalization > 0:
                    td_target /= normalization

                # Clip target to reasonable range (matching MAX_EVAL in engine)
                td_target = np.clip(td_target, -2000.0, 2000.0)

                # Add position and target to dataset
                positions.append(position["board"])
                td_targets.append(td_target)

        # Create dataset in the format expected by train_neural.py
        dataset = {
            "positions": [
                {"board": pos, "evaluation": target}
                for pos, target in zip(positions, td_targets)
            ]
        }

        # Save dataset
        with open(output_dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)

        print(
            f"Created TD(λ) dataset with {len(positions)} positions at {output_dataset_path}"
        )
        return output_dataset_path

    # Update output model paths
    def run_training_pipeline(self, start_iteration=1):
        """Run the complete training pipeline."""
        # Auto-detect latest iteration if starting from 1 (default)
        if start_iteration == 1:
            latest = self.find_latest_iteration()
            if latest >= 0:
                print(f"Detected existing models up to iteration {latest}.")
                print(f"Automatically continuing from iteration {latest + 1}.")
                start_iteration = latest + 1

        print(
            f"Starting TDChess training pipeline from iteration {start_iteration} (total iterations: {self.iterations})"
        )

        # Determine the model to start with
        if start_iteration == 1:
            self.ensure_initial_model()
            current_model = self.initial_model
        else:
            current_model = (
                self.model_dir / f"chess_model_iter_{start_iteration - 1}.pt"
            )
            if not current_model.exists():
                print(f"Warning: Starting model {current_model} not found!")
                print("Checking for any available model...")
                latest = self.find_latest_iteration()
                if latest >= 0:
                    current_model = self.model_dir / f"chess_model_iter_{latest}.pt"
                    print(f"Falling back to latest available model: {current_model}")
                    start_iteration = latest + 1
                else:
                    print("No models found. Starting from scratch.")
                    self.ensure_initial_model()
                    current_model = self.initial_model
                    start_iteration = 1

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
            # We pass current_model as initial_model to preserve weights (fine-tuning)
            train_model(
                str(td_dataset_path),
                str(output_model),
                100,
                batch_size=128,
                learning_rate=self.learning_rate,
                initial_model_path=str(current_model),
            )

            # Update current model for next iteration
            current_model = output_model
            print(f"Completed iteration {i}. New model: {current_model}")

        print(f"Training pipeline complete! Final model: {current_model}")
        return current_model


def main():
    """Main entry point for TDChess training pipeline."""
    default_model_dir = SCRIPT_DIR.parent / "model"
    parser = argparse.ArgumentParser(description="TDChess Training Pipeline")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(default_model_dir),
        help="Directory for models and datasets",
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of training iterations"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=2048,
        help="Number of self-play games per iteration",
    )
    parser.add_argument(
        "--lambda", dest="lambda_value", type=float, default=0.7, help="TD(λ) parameter"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for move selection"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00005,
        help="Learning rate for neural network training",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=1,
        help="Starting iteration (default: 1, will auto-detect latest if models exist)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Number of parallel workers for game generation",
    )
    parser.add_argument(
        "--initial-positions",
        type=int,
        default=100000,
        help="Number of positions for initial dataset",
    )
    parser.add_argument(
        "--initial-depth",
        type=int,
        default=4,
        help="Search depth for initial dataset generation",
    )

    args = parser.parse_args()

    pipeline = TDChessTraining(
        model_dir=args.model_dir,
        iterations=args.iterations,
        games_per_iteration=args.games,
        lambda_value=args.lambda_value,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        num_workers=args.parallel,
        initial_positions=args.initial_positions,
        initial_depth=args.initial_depth,
    )

    pipeline.run_training_pipeline(start_iteration=args.start_iter)


if __name__ == "__main__":
    main()
