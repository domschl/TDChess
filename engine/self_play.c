#include "self_play.h"
#include "board.h"
#include "movegen.h"
#include "neural.h"
#include "eval.h"
#include "python_binding.h"
#include "td_learning.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Generate and export self-play games to a JSON file
bool generate_self_play_games(const char *model_path, const char *output_path,
                              int num_games, float temperature) {
    printf("Generating %d self-play games with temperature %.2f\n",
           num_games, temperature);

    // Initialize neural network
    if (!initialize_neural(model_path)) {
        printf("Failed to initialize neural model from %s\n", model_path);
        return false;
    }

    // Set evaluation type to neural
    EvaluationType original_type = get_evaluation_type();
    set_evaluation_type(EVAL_NEURAL);

    // Open output file
    FILE *file = fopen(output_path, "w");
    if (!file) {
        printf("Failed to open file %s for writing\n", output_path);
        shutdown_neural();
        set_evaluation_type(original_type);
        return false;
    }

    // Write JSON header
    fprintf(file, "{\n");
    fprintf(file, "  \"games\": [\n");

    // Generate each game
    for (int game_idx = 0; game_idx < num_games; game_idx++) {
        // Start a new game
        fprintf(file, "    {\n");
        fprintf(file, "      \"positions\": [\n");

        // Set up the initial position
        Board board;
        setup_default_position(&board);

// Array to store game positions for later export
#define MAX_GAME_MOVES 500
        Board positions[MAX_GAME_MOVES];
        float evaluations[MAX_GAME_MOVES];
        int move_count = 0;

        // Record initial position
        memcpy(&positions[move_count], &board, sizeof(Board));
        evaluations[move_count] = evaluate_neural(&board) / 100.0f;  // Convert to pawn units
        move_count++;

        // Game result (default to draw)
        float game_result = 0.0f;

        // Play the game
        for (int move_num = 0; move_num < MAX_GAME_MOVES - 1; move_num++) {
            // Check for game end conditions
            MoveList moves;
            generate_legal_moves(&board, &moves);

            if (moves.count == 0) {
                // Checkmate or stalemate
                int king_square = -1;
                for (int sq = 0; sq < 64; sq++) {
                    if (board.pieces[sq].type == KING &&
                        board.pieces[sq].color == board.side_to_move) {
                        king_square = sq;
                        break;
                    }
                }

                if (king_square != -1 &&
                    is_square_attacked(&board, king_square, !board.side_to_move)) {
                    // Checkmate
                    game_result = (board.side_to_move == WHITE) ? -1.0f : 1.0f;
                    printf("Game %d: Checkmate! %s wins\n",
                           game_idx + 1, (board.side_to_move == WHITE) ? "Black" : "White");
                } else {
                    // Stalemate
                    game_result = 0.0f;
                    printf("Game %d: Stalemate! Draw\n", game_idx + 1);
                }
                break;
            }

            // Check for draw conditions
            if (board.halfmove_clock >= 100) {
                // Fifty-move rule
                printf("Game %d: Draw by fifty-move rule\n", game_idx + 1);
                break;
            }

            // Calculate temperature for this move (gradually decrease)
            float move_temperature = temperature *
                                     (1.0f - (float)move_num / MAX_GAME_MOVES * 0.7f);
            if (move_temperature < 0.1f) move_temperature = 0.1f;

            // Select move with temperature
            // (This would call your existing move selection function)
            Move move = select_move_with_randomness(&board, move_temperature);

            // Make the move
            make_move(&board, &move);

            // Record the position
            memcpy(&positions[move_count], &board, sizeof(Board));
            evaluations[move_count] = evaluate_neural(&board) / 100.0f;  // Convert to pawn units
            move_count++;
        }

        // Export all positions in this game
        for (int i = 0; i < move_count; i++) {
            // Start position object
            fprintf(file, "        {\n");

            // Export board representation
            char json_buffer[8192] = {0};
            if (!export_board_to_json(&positions[i], json_buffer, sizeof(json_buffer))) {
                printf("Failed to export board at position %d\n", i);
                continue;
            }

            fprintf(file, "          \"board\": %s,\n", json_buffer);
            fprintf(file, "          \"evaluation\": %.6f,\n", evaluations[i]);
            fprintf(file, "          \"side_to_move\": \"%s\"\n",
                    positions[i].side_to_move == WHITE ? "WHITE" : "BLACK");

            // End position object
            fprintf(file, "        }%s\n", (i < move_count - 1) ? "," : "");
        }

        // End positions array
        fprintf(file, "      ],\n");

        // Include game result
        fprintf(file, "      \"result\": %.1f\n", game_result);

        // End game object
        fprintf(file, "    }%s\n", (game_idx < num_games - 1) ? "," : "");

        // Progress update
        printf("Generated game %d/%d with %d moves\n",
               game_idx + 1, num_games, move_count - 1);
    }

    // Write JSON footer
    fprintf(file, "  ]\n");
    fprintf(file, "}\n");

    // Clean up
    fclose(file);
    shutdown_neural();
    set_evaluation_type(original_type);

    printf("Successfully exported %d games to %s\n", num_games, output_path);
    return true;
}
