from math import e
import os
import chess
import chess.engine
import json
import subprocess
import random
import sys
from typing import Any, Optional

from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.absolute()
STOCKFISH_PATHS = ["/opt/homebrew/bin/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"]  # IMPORTANT: Update this path
OUTPUT_DATASET_PATH = SCRIPT_DIR.parent / "model" / "initial_dataset.json"
NUM_POSITIONS_TO_GENERATE = 5000  # Adjust as needed
STOCKFISH_THINK_TIME = 0.1  # Seconds per evaluation
MAX_MOVES_FOR_RANDOM_POSITIONS = 30 # Max ply for generating diverse positions

# --- Helper Functions ---

def initialize_stockfish_engine(engine_paths: list[str]) -> chess.engine.SimpleEngine | None:
    """Initializes the Stockfish engine with multi-threading."""
    for engine_path in engine_paths:
        if os.path.exists(engine_path):
            try:
                engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                # Set options for faster processing
                engine.configure({"Threads": 8, "Hash": 1024}) 
                return engine
            except FileNotFoundError:
                pass
    print("Please update STOCKFISH_PATHS in the script.", flush=True)
    return None

def get_stockfish_evaluation(engine: chess.engine.SimpleEngine, board: chess.Board, think_time: float) -> Optional[float]:
    """Gets Stockfish evaluation for a given board state, from the current player's perspective."""
    try:
        info = engine.analyse(board, chess.engine.Limit(time=think_time))
        
        # Get score from White's perspective
        # This keeps it consistent with the engine's neural network targets
        score_obj = info.get("score")
        if score_obj is None:
             print(f"Warning: Stockfish returned no score for FEN {board.fen()}", flush=True)
             return None
             
        score_obj = score_obj.pov(chess.WHITE)

        if score_obj.is_mate():
            # Assign a large finite value for mate.
            mate_value = 100000 
            mate_ply = score_obj.mate()
            if mate_ply is None:
                 return 0.0 # Should not happen if is_mate() is true
                 
            if mate_ply > 0: # White mates
                 return mate_value - mate_ply 
            else: # Black mates
                 return -mate_value - mate_ply 
        else:
            # Clamp centipawn scores to a reasonable range, e.g., +-10000
            cp_eval = score_obj.score(mate_score=100000) 
            if cp_eval is None:
                 return 0.0
            return max(min(float(cp_eval), 10000.0), -10000.0)

    except Exception as e:
        print(f"Error getting evaluation for FEN {board.fen()}: {e}", flush=True)
        return None

def convert_board_to_tensor(board: chess.Board) -> list[float] | None:
    """
    Converts a chess.Board object to the 14x8x8 tensor representation (flattened),
    matching the logic in neural.c's board_to_planes.

    The 14 planes are:
    - Planes 0-5: White pieces (P, N, B, R, Q, K)
    - Planes 6-11: Black pieces (p, n, b, r, q, k)
    - Plane 12: Side to move (1.0 for White's turn, 0.0 for Black's turn, filled across the plane)
    - Plane 13: En passant square (1.0 at the en passant target square, 0.0 elsewhere)

    Output is a flat list of 14 * 8 * 8 = 896 floats (0.0 or 1.0).
    """
    INPUT_CHANNELS = 14  # As defined in neural.h
    BOARD_SIZE = 8       # As defined in neural.h
    tensor_size = INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE
    tensor = [0.0] * tensor_size

    # Planes 0-11: Piece positions
    # piece.piece_type in python-chess: PAWN=1, KNIGHT=2, ..., KING=6
    # piece_type_offset for C: PAWN=0, ..., KING=5
    for sq_idx in chess.SQUARES:  # Iterates 0 (A1) to 63 (H8)
        piece = board.piece_at(sq_idx)
        if piece:
            piece_type_offset = piece.piece_type - 1
            color_base_plane = 0 if piece.color == chess.WHITE else 6
            plane_idx = color_base_plane + piece_type_offset
            
            # CHW format: tensor_index = plane_idx * height * width + square_index
            # square_index (sq_idx) is 0-63, matching C's direct indexing
            tensor_idx = plane_idx * (BOARD_SIZE * BOARD_SIZE) + sq_idx
            tensor[tensor_idx] = 1.0

    # Plane 12: Side to move
    side_to_move_plane_idx = 12
    side_to_move_value = 1.0 if board.turn == chess.WHITE else 0.0
    plane_start_idx = side_to_move_plane_idx * (BOARD_SIZE * BOARD_SIZE)
    for i in range(BOARD_SIZE * BOARD_SIZE):
        tensor[plane_start_idx + i] = side_to_move_value

    # Plane 13: En passant square
    en_passant_plane_idx = 13
    if board.ep_square is not None:
        # board.ep_square is the index of the en passant target square (0-63)
        tensor_idx = en_passant_plane_idx * (BOARD_SIZE * BOARD_SIZE) + board.ep_square
        tensor[tensor_idx] = 1.0
    # Other squares in plane 13 remain 0.0 due to initialization.

    if len(tensor) != tensor_size:
        print(f"Error: Tensor has incorrect size {len(tensor)}. Expected {tensor_size} for FEN {board.fen()}.", flush=True)
        return None
        
    return tensor


def generate_diverse_positions(engine: chess.engine.SimpleEngine, num_positions: int, max_moves: int) -> list[chess.Board]:
    """Generates diverse chess positions by playing random legal moves followed by engine moves."""
    positions = []
    seen_fens = set()
    
    print(f"Generating {num_positions} unique positions with engine assisted diversity...", flush=True)
    
    attempts = 0
    while len(positions) < num_positions and attempts < num_positions * 10:
        attempts += 1
        board = chess.Board()
        
        # 1. Start with a few random moves (3-10) to diverge from the main line
        # Too many random moves lead to pure noise; 3-10 is a good "opening" range
        random_moves = random.randint(3, 10) 
        for _ in range(random_moves):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)
            
        # 2. Let the engine play a few moves (2-6) to reach a "reasonable" tactical state
        # This makes the positions more "chess-like" for the model to learn
        if not board.is_game_over():
            engine_moves = random.randint(2, 6)
            for _ in range(engine_moves):
                if board.is_game_over():
                    break
                # Fast play to get to a position
                result = engine.play(board, chess.engine.Limit(time=0.005))
                if result.move:
                    board.push(result.move)
        
        # 3. Deduplicate by FEN (ignoring move clocks/counts for broader matching)
        # We use the board part of the FEN to identify the piece configuration
        fen_key = board.epd() 
        if fen_key not in seen_fens:
            seen_fens.add(fen_key)
            positions.append(board.copy())
            
        if len(positions) % 100 == 0 and len(positions) > 0:
            print(f"  Found {len(positions)} unique positions... (Attempts: {attempts})", flush=True)
            
    return positions

# --- Main Script Logic ---
def main():
    print("Starting dataset generation with Stockfish...", flush=True)

    # Allow overriding number of positions via positional argument
    num_positions = NUM_POSITIONS_TO_GENERATE
    if len(sys.argv) > 1:
        try:
            num_positions = int(sys.argv[1])
            print(f"Overriding number of positions to generate: {num_positions}", flush=True)
        except ValueError:
            print(f"Invalid argument for number of positions: {sys.argv[1]}. Using default: {NUM_POSITIONS_TO_GENERATE}", flush=True)

    stockfish_engine = initialize_stockfish_engine(STOCKFISH_PATHS)
    if not stockfish_engine:
        return

    # This list will store individual position data dictionaries
    positions_data_list: list[dict[str, Any]] = []

    print(f"Generating {num_positions} diverse positions...", flush=True)
    chess_positions = generate_diverse_positions(stockfish_engine, num_positions, MAX_MOVES_FOR_RANDOM_POSITIONS)
    
    generated_count = 0
    for i, board in enumerate(chess_positions):
        if generated_count >= num_positions:
            break

        if i % 100 == 0:
            print(f"Processing position {i+1}/{len(chess_positions)} (Generated: {generated_count}) FEN: {board.fen()}", flush=True)

        evaluation = get_stockfish_evaluation(stockfish_engine, board, STOCKFISH_THINK_TIME)
        if evaluation is None:
            print(f"Skipping position due to evaluation error: {board.fen()}", flush=True)
            continue

        board_tensor = convert_board_to_tensor(board)
        if board_tensor is None:
            print(f"Skipping position due to tensor conversion error: {board.fen()}", flush=True)
            continue
        
        if len(board_tensor) != 896:
            print(f"Error: Tensor for FEN {board.fen()} has incorrect size {len(board_tensor)}. Expected 896.", flush=True)
            print("Please check your `convert_board_to_tensor` implementation.", flush=True)
            continue

        # Structure each position entry as expected by ChessDataset
        position_entry = {
            "board": {
                "tensor": board_tensor,
                "fen": board.fen()
            },
            "evaluation": evaluation
        }
        positions_data_list.append(position_entry)
        generated_count +=1

        if (i + 1) % 100 == 0:
            print(f"Generated {generated_count} positions so far...", flush=True)

    stockfish_engine.quit()
    print(f"Generated a total of {len(positions_data_list)} positions.", flush=True)

    if not positions_data_list:
        print("No data was generated. Exiting.", flush=True)
        return

    # Create the final dataset structure with the "positions" key
    final_dataset_to_save = {"positions": positions_data_list}

    print(f"Saving dataset to {OUTPUT_DATASET_PATH}...", flush=True)
    try:
        OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DATASET_PATH, 'w') as f:
            json.dump(final_dataset_to_save, f, indent=2) # Save the new structure
        print("Dataset saved successfully.", flush=True)
    except IOError as e:
        print(f"Error saving dataset: {e}")
        return

    print("\nStarting training process with the new dataset...", flush=True)
    try:
        # Ensure your Python environment for train_neural.py is active
        # or that it uses the correct interpreter.
        # You might need to specify the python executable if it's not in PATH
        # or if you are using virtual environments.
        # e.g., ['python3', 'train_neural.py', "--dataset", OUTPUT_DATASET_PATH]
        train_script_path = SCRIPT_DIR / "train_neural.py"
        training_command = ['python', str(train_script_path), "--dataset", str(OUTPUT_DATASET_PATH)]  
        print(f"Executing: {' '.join(training_command)}", flush=True)
        
        # It's often better to stream output or capture it,
        # but for simplicity, this will just run it.
        process = subprocess.Popen(training_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        if process.stdout:
            for line in iter(process.stdout.readline, b''):
                print(line.decode().strip(), flush=True)
            process.stdout.close()
        
        return_code = process.wait()

        if return_code == 0:
            print("Training process completed successfully.", flush=True)
        else:
            print(f"Training process failed with exit code {return_code}.", flush=True)
            print("Check the output above for errors from train_neural.py.", flush=True)

    except FileNotFoundError:
        print("Error: training/train_neural.py not found. Make sure it's in the correct path.", flush=True)
    except Exception as e:
        print(f"An error occurred while trying to run train_neural.py: {e}", flush=True)

if __name__ == "__main__":
    main()