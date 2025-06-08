import chess
import chess.engine
import json
import subprocess
import random
from typing import List, Dict, Any, Optional

# --- Configuration ---
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # IMPORTANT: Update this path
OUTPUT_DATASET_PATH = "model/initial_dataset.json"
NUM_POSITIONS_TO_GENERATE = 50000  # Adjust as needed
STOCKFISH_THINK_TIME = 0.1  # Seconds per evaluation
MAX_MOVES_FOR_RANDOM_POSITIONS = 30 # Max ply for generating diverse positions

# --- Helper Functions ---

def initialize_stockfish_engine(engine_path: str) -> Optional[chess.engine.SimpleEngine]:
    """Initializes the Stockfish engine."""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        return engine
    except FileNotFoundError:
        print(f"Error: Stockfish executable not found at {engine_path}")
        print("Please update STOCKFISH_PATH in the script.")
        return None
    except Exception as e:
        print(f"An error occurred while initializing Stockfish: {e}")
        return None

def get_stockfish_evaluation(engine: chess.engine.SimpleEngine, board: chess.Board, think_time: float) -> Optional[float]:
    """Gets Stockfish evaluation for a given board state, from the current player's perspective."""
    try:
        info = engine.analyse(board, chess.engine.Limit(time=think_time))
        
        # Get score from the perspective of the current player to move
        # board.turn == chess.WHITE means it's White's turn
        # board.turn == chess.BLACK means it's Black's turn
        score_obj = info["score"].pov(board.turn) 

        if score_obj.is_mate():
            # Assign a large finite value for mate.
            # For pov score, a positive mate score means the current player is mating.
            mate_value = 100000 
            # score_obj.mate() will be positive if the current player delivers mate.
            # We want to keep it positive and high.
            # If mate() is 1 (mate in 1 for current player), score should be high.
            # If mate() is -1 (mated in 1 by opponent), score should be very low.
            # The .pov() call already handles the perspective for mate scores.
            # A positive mate() from pov(board.turn) means board.turn has a mate.
            if score_obj.mate() > 0: # Current player mates
                 return mate_value - score_obj.mate() 
            else: # Current player is being mated (score_obj.mate() is negative)
                 return -mate_value - score_obj.mate() # Becomes a large negative
        else:
            # Clamp centipawn scores to a reasonable range, e.g., +-10000
            # score_obj.score() will give cp from current player's perspective
            cp_eval = score_obj.score(mate_score=100000) 
            return max(min(cp_eval, 10000.0), -10000.0)

    except Exception as e:
        print(f"Error getting evaluation for FEN {board.fen()}: {e}")
        return None

def convert_board_to_tensor(board: chess.Board) -> Optional[List[float]]:
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

    # Final check, though it should always be correct if logic is sound
    if len(tensor) != tensor_size:
        print(f"Error: Tensor has incorrect size {len(tensor)}. Expected {tensor_size} for FEN {board.fen()}.")
        return None
        
    return tensor


def generate_diverse_positions(num_positions: int, max_moves: int) -> List[chess.Board]:
    """Generates diverse chess positions by playing random legal moves."""
    positions = []
    for _ in range(num_positions):
        board = chess.Board()
        num_half_moves = random.randint(1, max_moves * 2) # Play up to max_moves full moves
        
        for _ in range(num_half_moves):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            random_move = random.choice(legal_moves)
            board.push(random_move)
        
        # Optional: Add some opening positions from a small book or list
        # Or, load positions from PGN files.
        
        positions.append(board.copy()) # Add a copy of the board state
    return positions

# --- Main Script Logic ---
def main():
    print("Starting dataset generation with Stockfish...")

    stockfish_engine = initialize_stockfish_engine(STOCKFISH_PATH)
    if not stockfish_engine:
        return

    # This list will store individual position data dictionaries
    positions_data_list: List[Dict[str, Any]] = []

    print(f"Generating {NUM_POSITIONS_TO_GENERATE} diverse positions...")
    chess_positions = generate_diverse_positions(NUM_POSITIONS_TO_GENERATE, MAX_MOVES_FOR_RANDOM_POSITIONS)
    
    generated_count = 0
    for i, board in enumerate(chess_positions):
        if generated_count >= NUM_POSITIONS_TO_GENERATE:
            break

        print(f"Processing position {i+1}/{len(chess_positions)} (Generated: {generated_count}) FEN: {board.fen()}")

        evaluation = get_stockfish_evaluation(stockfish_engine, board, STOCKFISH_THINK_TIME)
        if evaluation is None:
            print(f"Skipping position due to evaluation error: {board.fen()}")
            continue

        board_tensor = convert_board_to_tensor(board)
        if board_tensor is None:
            print(f"Skipping position due to tensor conversion error: {board.fen()}")
            continue
        
        if len(board_tensor) != 896:
            print(f"Error: Tensor for FEN {board.fen()} has incorrect size {len(board_tensor)}. Expected 896.")
            print("Please check your `convert_board_to_tensor` implementation.")
            continue

        # Structure each position entry as expected by ChessDataset
        position_entry = {
            "board": {
                "tensor": board_tensor
            },
            "evaluation": evaluation
        }
        positions_data_list.append(position_entry)
        generated_count +=1

        if (i + 1) % 100 == 0:
            print(f"Generated {generated_count} positions so far...")

    stockfish_engine.quit()
    print(f"Generated a total of {len(positions_data_list)} positions.")

    if not positions_data_list:
        print("No data was generated. Exiting.")
        return

    # Create the final dataset structure with the "positions" key
    final_dataset_to_save = {"positions": positions_data_list}

    print(f"Saving dataset to {OUTPUT_DATASET_PATH}...")
    try:
        with open(OUTPUT_DATASET_PATH, 'w') as f:
            json.dump(final_dataset_to_save, f, indent=2) # Save the new structure
        print("Dataset saved successfully.")
    except IOError as e:
        print(f"Error saving dataset: {e}")
        return

    print("\nStarting training process with the new dataset...")
    try:
        # Ensure your Python environment for train_neural.py is active
        # or that it uses the correct interpreter.
        # You might need to specify the python executable if it's not in PATH
        # or if you are using virtual environments.
        # e.g., ['python3', 'train_neural.py', OUTPUT_DATASET_PATH]
        training_command = ['python', 'train_neural.py', OUTPUT_DATASET_PATH]
        print(f"Executing: {' '.join(training_command)}")
        
        # It's often better to stream output or capture it,
        # but for simplicity, this will just run it.
        process = subprocess.Popen(training_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        if process.stdout:
            for line in iter(process.stdout.readline, b''):
                print(line.decode().strip())
            process.stdout.close()
        
        return_code = process.wait()

        if return_code == 0:
            print("Training process completed successfully.")
        else:
            print(f"Training process failed with exit code {return_code}.")
            print("Check the output above for errors from train_neural.py.")

    except FileNotFoundError:
        print("Error: train_neural.py not found. Make sure it's in the correct path.")
    except Exception as e:
        print(f"An error occurred while trying to run train_neural.py: {e}")

if __name__ == "__main__":
    main()