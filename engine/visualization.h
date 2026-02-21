#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "board.h"
#include "movegen.h"
#include <stdbool.h>

// Print a chess board with Unicode symbols and ANSI colors
void print_board_pretty(const Board *board);

// Convert a move to algebraic notation
char *move_to_algebraic(const Board *board, const Move *move, char *buffer, size_t buffer_size);

// Print a move in algebraic notation
void print_move_algebraic(const Board *board, const Move *move);

// Print a full game with positions at specified intervals
void print_game_debug(const Board *positions, int num_positions, int interval);

// Print a game with moves in algebraic notation and evaluations
void print_game_with_evals(const Board *positions, float *evaluations, int num_positions);

// Print positions with evaluations (simpler alternative to print_game_with_evals)
void print_positions_with_evals(const Board *positions, float *evaluations, int num_positions);

// Check if neural evaluation is available
bool is_neural_initialized(void);


// Print an evaluation score with infinity symbol for mates
void print_evaluation(float score);

// Format an evaluation score into a string
char* format_evaluation(float score, char* buffer, size_t size);

#endif  // VISUALIZATION_H
