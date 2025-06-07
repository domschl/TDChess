#ifndef TD_LEARNING_H
#define TD_LEARNING_H

#include "board.h"
#include <stdbool.h>

// Position with evaluation
typedef struct {
    Board board;
    float evaluation;
} GamePosition;

// Game structure for TD-Lambda learning
typedef struct {
    GamePosition *positions;
    int move_count;
    float game_result;  // 1.0 for white win, 0.0 for draw, -1.0 for black win
} Game;

// TD-Lambda learning parameters
typedef struct {
    float lambda;             // Lambda parameter (typically 0.7-0.9)
    float learning_rate;      // Learning rate for updates
    float temperature;        // Temperature for move selection randomness
    int num_games;            // Number of games to generate
    int max_moves;            // Maximum moves per game
    const char *model_path;   // Current model path
    const char *output_path;  // Output dataset path
} TDLambdaParams;

// Generate self-play games using TD-Lambda learning
bool generate_td_lambda_dataset(const TDLambdaParams *params);

// Calculate TD targets and export dataset
bool export_td_lambda_dataset(Game *games, int num_games, const TDLambdaParams *params);

// Function to check if position is quiet (for search)
bool is_position_quiet(const Board *board);

#endif  // TD_LEARNING_H
