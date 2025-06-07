#ifndef TD_LEARNING_H
#define TD_LEARNING_H

#include "board.h"
#include "movegen.h"

// Parameters for TD-Lambda learning
typedef struct {
    const char *model_path;   // Path to neural network model
    const char *output_path;  // Path to save dataset
    int num_games;            // Number of self-play games to generate
    int max_moves;            // Maximum moves per game
    float lambda;             // TD-Lambda parameter
    float temperature;        // Temperature for move selection
    float learning_rate;      // Learning rate for updates
} TDLambdaParams;

// Structure to hold game positions and associated data
typedef struct {
    Board board;       // Current board state
    Board prev_board;  // Previous board state (for move reconstruction)
    float evaluation;  // Neural evaluation of position
    Move last_move;    // Store the move that led to this position
    bool has_move;     // Flag to indicate if last_move is valid
} GamePosition;

// Structure to hold a complete game
typedef struct {
    GamePosition *positions;
    int move_count;
    float game_result;  // 1.0 for white win, -1.0 for black win, 0.0 for draw
} Game;

// Generate TD-Lambda dataset from self-play
bool generate_td_lambda_dataset(const TDLambdaParams *params);

#endif  // TD_LEARNING_H
