#include "self_play.h"
#include "board.h"
#include "movegen.h"
#include "neural.h"
#include "eval.h"
#include "python_binding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Generate and export self-play games to a JSON file
bool generate_self_play_games(const char *model_path, const char *output_path,
                              int num_games, float temperature, unsigned int seed) {
    printf("Generating %d self-play games with temperature %.2f (seed %u)\n",
           num_games, temperature, seed);

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
        // Seed each game individually to ensure diversity, especially in parallel workers
        // Using both base seed and game index to ensure unique sequences
        srand(seed + game_idx);

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

// Random move selection to increase diversity
Move select_move_with_randomness(Board *board, float temperature) {
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

    // Add safety check for obvious blunders
    // First pass: identify if any moves are losing material immediately
    bool has_non_blunder_moves = false;
    bool is_blunder[MAX_MOVES] = {0};

    // Piece values for basic material counting
    const int piece_values[7] = {0, 100, 320, 330, 500, 900, 20000};  // EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

    for (int i = 0; i < moves.count; i++) {
        // Check if move puts a piece on a square attacked by a lower-value piece
        int to_square = moves.moves[i].to;
        int from_square = moves.moves[i].from;
        PieceType moved_piece = board->pieces[from_square].type;

        // Skip pawn promotions - these are usually good
        if (moved_piece == PAWN && (to_square >= 56 || to_square <= 7)) {
            has_non_blunder_moves = true;
            continue;
        }

        // Make the move to analyze the resulting position
        make_move(board, &moves.moves[i]);

        // Now check if the moved piece can be captured
        bool is_hanging = false;
        int moved_piece_value = piece_values[moved_piece];
        int lowest_attacker_value = 20000;  // Initialize to king value

        // Check if any opponent pieces attack the destination square
        for (int sq = 0; sq < 64; sq++) {
            if (board->pieces[sq].type != EMPTY &&
                board->pieces[sq].color == board->side_to_move) {

                // Check if this piece attacks our moved piece
                Board test_board = *board;
                // Fix: Initialize all fields in the Move struct
                Move capture = {
                    .from = sq,
                    .to = to_square,
                    .promotion = EMPTY,
                    .capture = true,
                    .castling = false,
                    .en_passant = false,
                    .captured_piece_square = to_square,
                    .captured_piece_type = board->pieces[to_square].type,
                    .captured_piece_color = board->pieces[to_square].color,
                    .old_castle_rights = test_board.castle_rights,
                    .old_en_passant = test_board.en_passant_square,
                    .old_halfmove_clock = test_board.halfmove_clock};

                if (is_move_legal(&test_board, capture)) {
                    is_hanging = true;
                    int attacker_value = piece_values[board->pieces[sq].type];
                    if (attacker_value < lowest_attacker_value) {
                        lowest_attacker_value = attacker_value;
                    }
                }
            }
        }

        // Unmake the move
        unmake_move(board, moves.moves[i]);

        // If the piece is hanging and it's a bad trade, mark as blunder
        if (is_hanging && lowest_attacker_value < moved_piece_value) {
            is_blunder[i] = true;
        } else {
            has_non_blunder_moves = true;
        }
    }

    // Evaluate all moves
    float scores[MAX_MOVES];
    float max_score = -INFINITY;
    float min_score = INFINITY;

    for (int i = 0; i < moves.count; i++) {
        // Skip obvious blunders if we have better moves
        if (is_blunder[i] && has_non_blunder_moves) {
            scores[i] = -INFINITY;
            continue;
        }

        make_move(board, &moves.moves[i]);

        // Negate the evaluation to get it from our perspective
        scores[i] = -evaluate_neural(board);

        if (scores[i] > max_score) max_score = scores[i];
        if (scores[i] < min_score) min_score = scores[i];

        unmake_move(board, moves.moves[i]);
    }

    // Epsilon-greedy noise: with small probability, pick a completely random non-blunder move
    float epsilon = 0.05f; 
    if (((float)rand() / RAND_MAX) < epsilon) {
        int non_blunder_count = 0;
        int non_blunder_indices[MAX_MOVES];
        for (int i = 0; i < moves.count; i++) {
            if (!is_blunder[i] || !has_non_blunder_moves) {
                non_blunder_indices[non_blunder_count++] = i;
            }
        }
        if (non_blunder_count > 0) {
            return moves.moves[non_blunder_indices[rand() % non_blunder_count]];
        }
    }

    // Apply temperature and convert to probabilities
    // Temperature is typically in "pawn units" (e.g. 0.8), but scores are in centipawns.
    // We scale temperature by 100 to match centipawns.
    float scaled_temp = temperature * 100.0f;
    if (scaled_temp < 1.0f) scaled_temp = 1.0f; // Prevent division by very small/zero

    float total_probability = 0.0f;
    for (int i = 0; i < moves.count; i++) {
        if (scores[i] == -INFINITY) {
            continue;
        }

        // Adjust scores relative to maximum (for numerical stability)
        scores[i] = exp((scores[i] - max_score) / scaled_temp);
        total_probability += scores[i];
    }

    // Choose move based on probabilities
    float choice = ((float)rand() / RAND_MAX) * total_probability;
    float cumulative = 0.0f;

    for (int i = 0; i < moves.count; i++) {
        if (scores[i] == -INFINITY) continue;

        cumulative += scores[i];
        if (choice <= cumulative) {
            return moves.moves[i];
        }
    }

    // Fallback - find the best non-blunder move
    float best_score = -INFINITY;
    int best_idx = 0;

    for (int i = 0; i < moves.count; i++) {
        if (!is_blunder[i] && scores[i] > best_score) {
            best_score = scores[i];
            best_idx = i;
        }
    }

    return moves.moves[best_idx];
}
