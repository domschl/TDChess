#!/usr/bin/env python3
"""
Evaluate a specific FEN using a trained TDChess model.
Usage: python evaluate_fen.py --fen "FEN" --model "path_to_model.pt"
"""
import argparse
import torch
import chess
import json
import os
from pathlib import Path
from train_neural import ChessNet, ChessDataset

SCRIPT_DIR = Path(__file__).parent.absolute()

def convert_fen_to_tensor(fen: str):
    """Converts a FEN to the 14x8x8 tensor representation."""
    board = chess.Board(fen)
    INPUT_CHANNELS = 14
    BOARD_SIZE = 8
    tensor = torch.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE))

    # Planes 0-5: White pieces
    # Planes 6-11: Black pieces
    for sq_idx in chess.SQUARES:
        piece = board.piece_at(sq_idx)
        if piece:
            piece_type_offset = piece.piece_type - 1
            color_base_plane = 0 if piece.color == chess.WHITE else 6
            plane_idx = color_base_plane + piece_type_offset
            
            # sq_idx is 0-63 (A1 to H8)
            rank = sq_idx // 8
            file = sq_idx % 8
            tensor[plane_idx, rank, file] = 1.0

    # Plane 12: Side to move
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Plane 13: En passant square
    if board.ep_square is not None:
        rank = board.ep_square // 8
        file = board.ep_square % 8
        tensor[13, rank, file] = 1.0
        
    return tensor.unsqueeze(0) # Add batch dimension

def main():
    parser = argparse.ArgumentParser(description='Evaluate a FEN using a trained TDChess model')
    parser.add_argument('--fen', type=str, default=chess.STARTING_FEN, help='FEN to evaluate')
    parser.add_argument('--model', type=str, required=True, help='Path to trained PyTorch model')
    parser.add_argument('--max-eval', type=float, default=2000.0, help='Max eval used during training')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        return

    # Load model
    device = torch.device("cpu")
    model = ChessNet()
    
    try:
        checkpoint = torch.load(args.model, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            if "max_eval" in checkpoint:
                args.max_eval = checkpoint["max_eval"]
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {args.model}")
        print(f"Using max_eval: {args.max_eval}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # Evaluate
    input_tensor = convert_fen_to_tensor(args.fen)
    with torch.no_grad():
        output = model(input_tensor)
    
    # The output is normalized [-1, 1] due to the tanh in forward()
    normalized_score = output.item()
    centipawns = normalized_score * args.max_eval
    
    print(f"\nFEN: {args.fen}")
    print(f"Normalized score (White's perspective): {normalized_score:.4f}")
    print(f"Evaluation (White's perspective): {centipawns:.2f} cp")
    
    # Perspective relative to side to move
    board = chess.Board(args.fen)
    side_to_move_eval = centipawns if board.turn == chess.WHITE else -centipawns
    print(f"Evaluation (Side to move): {side_to_move_eval:.2f} cp")

if __name__ == '__main__':
    main()
