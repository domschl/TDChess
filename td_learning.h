#ifndef TD_LEARNING_H
#define TD_LEARNING_H

#include "board.h"
#include <stdbool.h>

// TD-Lambda learning parameters
typedef struct {
    float lambda;             // Lambda parameter (typically 0.7-0.9)
    float learning_rate;      // Learning rate for updates
    int num_games;            // Number of games to generate
    int max_moves;            // Maximum moves per game
    const char *model_path;   // Current model path
    const char *output_path;  // Output model path
} TDLambdaParams;

// Generate self-play games using TD-Lambda learning
bool generate_td_lambda_dataset(const TDLambdaParams *params);

// Update model using TD-Lambda
bool train_td_lambda(const TDLambdaParams *params);

#endif  // TD_LEARNING_H
