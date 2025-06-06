#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "board.h"
#include <stddef.h>

// Maximum number of possible moves from any position
#define MAX_MOVES 256

// Move list structure
typedef struct {
    Move moves[MAX_MOVES];
    int scores[MAX_MOVES];  // Add scores for move ordering
    int count;
} MoveList;

// Move generation functions
void generate_moves(const Board *board, MoveList *list);
void generate_legal_moves(const Board *board, MoveList *list);
void generate_captures(const Board *board, MoveList *list);
void generate_pawn_captures(const Board *board, MoveList *list);
void generate_knight_captures(const Board *board, MoveList *list);
void generate_bishop_captures(const Board *board, MoveList *list);
void generate_rook_captures(const Board *board, MoveList *list);
void generate_queen_captures(const Board *board, MoveList *list);
void generate_king_captures(const Board *board, MoveList *list);

// Helper functions for specific piece types
void generate_pawn_moves(const Board *board, MoveList *list);
void generate_knight_moves(const Board *board, MoveList *list);
void generate_bishop_moves(const Board *board, MoveList *list);
void generate_rook_moves(const Board *board, MoveList *list);
void generate_queen_moves(const Board *board, MoveList *list);
void generate_king_moves(const Board *board, MoveList *list);

// Helper functions
bool is_move_legal(const Board *board, Move move);
void add_move(MoveList *list, int from, int to, PieceType promotion, bool capture, bool castling, bool en_passant, int captured_square);

// Move execution functions
void make_move(Board *board, Move move);
void unmake_move(Board *board, Move move);

// Move notation
char *move_to_string(Move move);
Move string_to_move(const Board *board, const char *str);

#endif  // MOVEGEN_H
