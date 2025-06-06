#include "td_learning.h"
#include "board.h"
#include "movegen.h"
#include "search.h"
#include "neural.h"
#include "eval.h"
#include "python_binding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef struct {
    Board board;
    float evaluation;
} GamePosition;

typedef struct {
    GamePosition *positions;
    int move_count;
    float game_result;  // 1.0 for white win, 0.0 for draw, -1.0 for black win
} Game;

// Generate a single self-play game
static Game generate_self_play_game(const TDLambdaParams *params) {
    // Allocate memory for game positions
    Game game;
    game.positions = (GamePosition *)malloc(params->max_moves * sizeof(GamePosition));
    game.move_count = 0;
    game.game_result = 0.0f;  // Default to draw

    if (!game.positions) {
        printf("Failed to allocate memory for game positions\n");
        return game;
    }

    // Set up the initial position
    Board board;
    setup_default_position(&board);

    // Use neural evaluation for move generation
    EvaluationType original_type = get_evaluation_type();
    set_evaluation_type(EVAL_NEURAL);

    // Play the game
    for (int move_num = 0; move_num < params->max_moves; move_num++) {
        // Record the current position
        memcpy(&game.positions[game.move_count].board, &board, sizeof(Board));
        game.positions[game.move_count].evaluation = evaluate_neural(&board);
        game.move_count++;

        // Check for game end conditions
        MoveList moves;
        generate_legal_moves(&board, &moves);

        if (moves.count == 0) {
            // Checkmate or stalemate
            int king_square = -1;
            for (int sq = 0; sq < 64; sq++) {
                if (board.pieces[sq].type == KING && board.pieces[sq].color == board.side_to_move) {
                    king_square = sq;
                    break;
                }
            }

            if (king_square != -1 && is_square_attacked(&board, king_square, !board.side_to_move)) {
                // Checkmate
                game.game_result = (board.side_to_move == WHITE) ? -1.0f : 1.0f;
            }
            break;
        }

        // Check for draw conditions
        if (board.halfmove_clock >= 100) {
            // Fifty-move rule
            break;
        }

        // Find and make the best move
        Move best_move;
        uint64_t nodes = 0;
        find_best_move(&board, 3, &best_move, &nodes);
        make_move(&board, &best_move);
    }

    // Restore original evaluation type
    set_evaluation_type(original_type);

    return game;
}

// Calculate TD errors and export training data
static bool export_td_lambda_dataset(Game *games, int num_games, const TDLambdaParams *params) {
    // Calculate how many total positions we have
    int total_positions = 0;
    for (int i = 0; i < num_games; i++) {
        total_positions += games[i].move_count;
    }

    // Allocate arrays for positions and TD targets
    Board *positions = (Board *)malloc(total_positions * sizeof(Board));
    float *td_targets = (float *)malloc(total_positions * sizeof(float));

    if (!positions || !td_targets) {
        printf("Failed to allocate memory for TD dataset\n");
        if (positions) free(positions);
        if (td_targets) free(td_targets);
        return false;
    }

    // Calculate TD targets for each position using TD(λ)
    int pos_index = 0;
    for (int game_idx = 0; game_idx < num_games; game_idx++) {
        Game *game = &games[game_idx];

        // Calculate TD targets for this game
        for (int move = 0; move < game->move_count; move++) {
            // Copy the position
            memcpy(&positions[pos_index], &game->positions[move].board, sizeof(Board));

            // Calculate TD target using the lambda parameter
            float td_target = 0.0f;
            float lambda_power = 1.0f;
            float normalization = 0.0f;

            // Look ahead up to the end of the game
            for (int future = move + 1; future < game->move_count; future++) {
                // int _steps = future - move;  // Calculates temporal distance
                lambda_power *= params->lambda;

                // Add this future evaluation to our target, weighted by lambda
                float future_eval = game->positions[future].evaluation;
                td_target += lambda_power * future_eval;
                normalization += lambda_power;
            }

            // Add the final game result with remaining lambda weight
            lambda_power *= params->lambda;
            td_target += lambda_power * game->game_result * 100.0f;  // Scale result to centipawns
            normalization += lambda_power;

            // Normalize the target
            if (normalization > 0.0f) {
                td_target /= normalization;
            } else {
                td_target = game->positions[move].evaluation;
            }

            // Store the TD target
            td_targets[pos_index] = td_target;
            pos_index++;
        }
    }

    // Export the dataset
    bool success = export_positions_to_dataset(params->output_path, positions, td_targets, total_positions);

    // Clean up
    free(positions);
    free(td_targets);

    return success;
}

// Generate self-play games and create TD-Lambda dataset
bool generate_td_lambda_dataset(const TDLambdaParams *params) {
    printf("Generating TD-Lambda dataset with %d games (lambda=%.2f)\n",
           params->num_games, params->lambda);

    // Initialize neural network with current model
    if (!initialize_neural(params->model_path)) {
        printf("Failed to initialize neural model from %s\n", params->model_path);
        return false;
    }

    // Generate self-play games
    Game *games = (Game *)malloc(params->num_games * sizeof(Game));
    if (!games) {
        printf("Failed to allocate memory for games\n");
        shutdown_neural();
        return false;
    }

    // Generate each game
    for (int i = 0; i < params->num_games; i++) {
        games[i] = generate_self_play_game(params);
        printf("Generated game %d/%d with %d moves\n", i + 1, params->num_games, games[i].move_count);
    }

    // Export TD-Lambda dataset
    bool success = export_td_lambda_dataset(games, params->num_games, params);

    // Clean up
    for (int i = 0; i < params->num_games; i++) {
        free(games[i].positions);
    }
    free(games);
    shutdown_neural();

    return success;
}
