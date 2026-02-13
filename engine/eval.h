#ifndef EVAL_H
#define EVAL_H

#include "board.h"

// Evaluation types
typedef enum {
    EVAL_BASIC,  // Simple material+position
    EVAL_NEURAL  // Neural network-based
} EvaluationType;

// Piece values in centipawns (100 = 1 pawn)
#define PAWN_VALUE 100
#define KNIGHT_VALUE 320
#define BISHOP_VALUE 330
#define ROOK_VALUE 500
#define QUEEN_VALUE 900
#define KING_VALUE 20000  // Not technically used for material counting

// Evaluation function types
typedef float (*EvaluationFunction)(const Board *board);
typedef bool (*QuiescenceCheckFunction)(const Board *board);

// Core evaluation functions
float evaluate_position(const Board *board);
bool is_position_quiet(const Board *board);

// Basic evaluation implementations
float evaluate_basic(const Board *board);
bool is_quiet_basic(const Board *board);

// Neural evaluation stubs (to be implemented later)
float evaluate_neural(const Board *board);
bool is_quiet_neural(const Board *board);

// Get/set evaluation type
EvaluationType get_evaluation_type(void);
void set_evaluation_type(EvaluationType type);

// Testing/utility functions
void print_evaluation_details(const Board *board);

#endif  // EVAL_H
