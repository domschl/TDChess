#include "td_learning.h"
#include "board.h"
#include "movegen.h"
#include "search.h"
#include "neural.h"
#include "eval.h"
#include "visualization.h"
#include "python_binding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Rename the function to avoid conflicts with board.c
static void validate_self_play_board_state(Board *board, int move_number) {
    // Fix negative or zero fullmove numbers
    if (board->fullmove_number < 1) {
        // Estimate the correct fullmove number based on the move count
        int estimated_fullmove = (move_number / 2) + 1;
        printf("Fixing invalid fullmove number: %d → %d\n",
               board->fullmove_number, estimated_fullmove);
        board->fullmove_number = estimated_fullmove;
    }

    // Ensure valid side to move
    if (board->side_to_move != WHITE && board->side_to_move != BLACK) {
        printf("Fixing invalid side to move\n");
        // Toggle from previous value, defaulting to WHITE if completely invalid
        board->side_to_move = (board->side_to_move == BLACK) ? WHITE : BLACK;
    }

    // Validate en passant square
    if (board->en_passant_square != -1) {
        int rank = SQUARE_RANK(board->en_passant_square);
        if ((board->side_to_move == WHITE && rank != 5) ||
            (board->side_to_move == BLACK && rank != 2)) {
            printf("Fixing invalid en passant square\n");
            board->en_passant_square = -1;
        }
    }
}

// Random move selection to increase diversity
static Move select_move_with_randomness(Board *board, float temperature) {
    // Generate all legal moves
    MoveList moves;
    generate_legal_moves(board, &moves);

    if (moves.count == 0) {
        // No legal moves - shouldn't happen as caller should check
        Move dummy = {0};
        return dummy;
    }

    if (moves.count == 1) {
        // Only one move, just return it
        return moves.moves[0];
    }

    // Evaluate all moves
    float scores[MAX_MOVES];
    float max_score = -INFINITY;
    float min_score = INFINITY;

    for (int i = 0; i < moves.count; i++) {
        make_move(board, &moves.moves[i]);

        // Score is negative of evaluation (since we're looking from opponent's view)
        scores[i] = -evaluate_neural(board);

        // Track min/max for normalization
        if (scores[i] > max_score) max_score = scores[i];
        if (scores[i] < min_score) min_score = scores[i];

        unmake_move(board, moves.moves[i]);
    }

    // Apply temperature and convert to probabilities
    float total_probability = 0.0f;
    for (int i = 0; i < moves.count; i++) {
        // Normalize score to [0,1] range then apply temperature
        float normalized_score = (scores[i] - min_score) / (max_score - min_score + 1e-6f);
        scores[i] = expf(normalized_score / temperature);
        total_probability += scores[i];
    }

    // Choose move based on probabilities
    float choice = ((float)rand() / RAND_MAX) * total_probability;
    float cumulative = 0.0f;

    for (int i = 0; i < moves.count; i++) {
        cumulative += scores[i];
        if (cumulative >= choice) {
            return moves.moves[i];
        }
    }

    // Fallback - shouldn't reach here
    return moves.moves[0];
}

// Generate a single self-play game with randomness and draw detection
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

    // Temperature schedule - start higher, decrease over time
    float base_temperature = params->temperature;

    // Play the game
    for (int move_num = 0; move_num < params->max_moves; move_num++) {
        // Validate board state before recording
        validate_self_play_board_state(&board, move_num);

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

        // Check for draw by repetition
        bool repetition_draw = false;
        for (int i = 0; i < game.move_count - 2; i++) {
            // Only check positions with the same side to move
            if (game.positions[i].board.side_to_move != board.side_to_move) {
                continue;
            }

            // Compare board positions
            bool same_position = true;
            for (int sq = 0; sq < 64; sq++) {
                if (game.positions[i].board.pieces[sq].type != board.pieces[sq].type ||
                    game.positions[i].board.pieces[sq].color != board.pieces[sq].color) {
                    same_position = false;
                    break;
                }
            }

            // If all pieces match and castling/en passant rights match
            if (same_position &&
                game.positions[i].board.castle_rights == board.castle_rights &&
                game.positions[i].board.en_passant_square == board.en_passant_square) {

                // Count occurrences of this position
                int count = 1;  // This position already
                for (int j = i + 1; j < game.move_count; j++) {
                    // Only check positions with the same side to move
                    if (game.positions[j].board.side_to_move != board.side_to_move) {
                        continue;
                    }

                    // Compare positions
                    bool match = true;
                    for (int sq = 0; sq < 64; sq++) {
                        if (game.positions[j].board.pieces[sq].type != board.pieces[sq].type ||
                            game.positions[j].board.pieces[sq].color != board.pieces[sq].color) {
                            match = false;
                            break;
                        }
                    }

                    if (match &&
                        game.positions[j].board.castle_rights == board.castle_rights &&
                        game.positions[j].board.en_passant_square == board.en_passant_square) {
                        count++;
                    }
                }

                if (count >= 2) {  // This is the third occurrence
                    repetition_draw = true;
                    break;
                }
            }
        }

        if (repetition_draw) {
            printf("Draw by threefold repetition\n");
            break;
        }

        // Calculate temperature for this move (gradually decrease)
        float temperature = base_temperature * (1.0f - (float)move_num / params->max_moves * 0.7f);
        if (temperature < 0.1f) temperature = 0.1f;  // Minimum temperature

        // Select move with randomness (without depth parameter)
        Move move = select_move_with_randomness(&board, temperature);

        // Make the move
        make_move(&board, &move);

        // Validate board state after making the move
        validate_self_play_board_state(&board, move_num + 1);
    }

    // Restore original evaluation type
    set_evaluation_type(original_type);

    return game;
}

// Calculate TD errors and export training data
bool export_td_lambda_dataset(Game *games, int num_games, const TDLambdaParams *params) {
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
    printf("Generating TD-Lambda dataset with %d games (lambda=%.2f, temp=%.2f)\n",
           params->num_games, params->lambda, params->temperature);

    // Seed random number generator
    srand((unsigned int)time(NULL));

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

        // Debug output: show one game in detail every N games
        if (i % 10 == 0 || params->num_games < 5) {
            printf("\n=== Detailed view of game %d ===\n", i + 1);

            // Create arrays for visualization
            Board *positions = malloc(games[i].move_count * sizeof(Board));
            float *evaluations = malloc(games[i].move_count * sizeof(float));

            for (int j = 0; j < games[i].move_count; j++) {
                positions[j] = games[i].positions[j].board;
                evaluations[j] = games[i].positions[j].evaluation;
            }

            // Try both visualization methods
            print_game_with_evals(positions, evaluations, games[i].move_count);
            printf("\n=== Alternative view (all positions) ===\n");
            print_positions_with_evals(positions, evaluations, games[i].move_count);

            free(positions);
            free(evaluations);
        }
    }

    // Export TD-Lambda dataset
    printf("Exporting dataset to %s\n", params->output_path);
    bool success = export_td_lambda_dataset(games, params->num_games, params);

    // Clean up
    for (int i = 0; i < params->num_games; i++) {
        free(games[i].positions);
    }
    free(games);
    shutdown_neural();

    return success;
}
