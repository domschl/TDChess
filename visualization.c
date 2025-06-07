#include "visualization.h"
#include "eval.h"
#include "neural.h"  // Added for get_neural_evaluator()
#include <stdio.h>
#include <string.h>

// ANSI color codes
#define ANSI_RESET "\x1B[0m"
#define ANSI_BLACK "\x1B[30m"
#define ANSI_RED "\x1B[31m"
#define ANSI_GREEN "\x1B[32m"
#define ANSI_YELLOW "\x1B[33m"
#define ANSI_BLUE "\x1B[34m"
#define ANSI_MAGENTA "\x1B[35m"
#define ANSI_CYAN "\x1B[36m"
#define ANSI_WHITE "\x1B[37m"
#define ANSI_BG_WHITE "\x1B[47m"
#define ANSI_BG_BLACK "\x1B[40m"
#define ANSI_BG_GRAY "\x1B[100m"
#define ANSI_BG_BRIGHT_WHITE "\x1B[107m"
#define ANSI_BRIGHT_BLACK "\x1B[90m"
#define ANSI_BRIGHT_WHITE "\x1B[97m"

// Unicode chess symbols
const char *UNICODE_PIECES[7][2] = {
    {"", ""},    // EMPTY
    {"♙", "♟"},  // PAWN
    {"♘", "♞"},  // KNIGHT
    {"♗", "♝"},  // BISHOP
    {"♖", "♜"},  // ROOK
    {"♕", "♛"},  // QUEEN
    {"♔", "♚"}   // KING
};

// Print a chess board with Unicode symbols and ANSI colors
void print_board_pretty(const Board *board) {
    printf("\n");

    // Print top border with file labels
    printf("    ");
    for (int file = 0; file < 8; file++) {
        printf(" %c  ", 'a' + file);  // Adjusted spacing
    }
    printf("\n");

    // Print board
    for (int rank = 7; rank >= 0; rank--) {
        printf(" %d ", rank + 1);

        for (int file = 0; file < 8; file++) {
            int square = SQUARE(file, rank);
            Piece piece = board->pieces[square];

            // Determine square color
            const char *bg_color = ((file + rank) % 2 == 0) ? ANSI_BG_WHITE : ANSI_BG_BRIGHT_WHITE;
            const char *text_color = (piece.color == WHITE) ? ANSI_YELLOW : ANSI_BLUE;

            // Print piece with colors - ensure consistent width
            if (piece.type == EMPTY) {
                // Empty square - three spaces for consistent width
                printf("%s    %s", bg_color, ANSI_RESET);
            } else {
                // Square with piece - space, piece, space
                printf("%s%s  %s %s", bg_color, text_color,
                       UNICODE_PIECES[piece.type][piece.color], ANSI_RESET);
            }
        }

        printf(" %d\n", rank + 1);
    }

    // Print bottom file labels
    printf("    ");
    for (int file = 0; file < 8; file++) {
        printf(" %c  ", 'a' + file);  // Adjusted spacing
    }
    printf("\n");

    if (false) {
        // Print side to move
        printf("\nSide to move: %s\n", board->side_to_move == WHITE ? "White" : "Black");

        // Print castling rights
        printf("Castling: ");
        if (board->castle_rights & CASTLE_WHITE_KINGSIDE) printf("K");
        if (board->castle_rights & CASTLE_WHITE_QUEENSIDE) printf("Q");
        if (board->castle_rights & CASTLE_BLACK_KINGSIDE) printf("k");
        if (board->castle_rights & CASTLE_BLACK_QUEENSIDE) printf("q");
        if (board->castle_rights == 0) printf("-");
        printf("\n");

        // Print en passant square
        printf("En passant: ");
        if (board->en_passant_square != -1) {
            int file = SQUARE_FILE(board->en_passant_square);
            int rank = SQUARE_RANK(board->en_passant_square);
            printf("%c%d", 'a' + file, rank + 1);
        } else {
            printf("-");
        }
        printf("\n");

        // Print halfmove clock and fullmove number
        printf("Halfmove clock: %d\n", board->halfmove_clock);
        printf("Fullmove number: %d\n", board->fullmove_number);
    }
}

// Convert a move to algebraic notation
char *move_to_algebraic(const Board *board, const Move *move, char *buffer, size_t buffer_size) {
    if (!move || !buffer || buffer_size < 10) return NULL;  // Increased minimum buffer size for Unicode

    // Special case for castling
    if (move->castling) {
        if (SQUARE_FILE(move->to) > SQUARE_FILE(move->from)) {
            snprintf(buffer, buffer_size, "O-O");
        } else {
            snprintf(buffer, buffer_size, "O-O-O");
        }
        return buffer;
    }

    // Determine piece symbol - use Unicode chess pieces
    const char *piece_symbol = "";
    int from_piece_type = board->pieces[move->from].type;
    int piece_color = board->pieces[move->from].color;

    // Use the appropriate Unicode piece based on type and color
    if (from_piece_type != PAWN) {
        piece_symbol = UNICODE_PIECES[from_piece_type][piece_color];
    }

    // Source and destination coordinates
    char src_file = 'a' + SQUARE_FILE(move->from);
    char dst_file = 'a' + SQUARE_FILE(move->to);
    char dst_rank = '1' + SQUARE_RANK(move->to);

    // Basic move notation
    if (from_piece_type == PAWN) {
        if (move->capture) {
            snprintf(buffer, buffer_size, "%cx%c%c", src_file, dst_file, dst_rank);
        } else {
            snprintf(buffer, buffer_size, "%c%c", dst_file, dst_rank);
        }
    } else {
        if (move->capture) {
            snprintf(buffer, buffer_size, "%sx%c%c", piece_symbol, dst_file, dst_rank);
        } else {
            snprintf(buffer, buffer_size, "%s%c%c", piece_symbol, dst_file, dst_rank);
        }
    }

    // Add promotion piece if applicable
    if (move->promotion != EMPTY) {
        const char *promotion_symbol = UNICODE_PIECES[move->promotion][piece_color];

        size_t len = strlen(buffer);
        if (len + strlen(promotion_symbol) + 1 < buffer_size) {
            buffer[len] = '=';
            strcpy(&buffer[len + 1], promotion_symbol);
        }
    }

    return buffer;
}

// Print a move in algebraic notation
void print_move_algebraic(const Board *board, const Move *move) {
    char buffer[8];
    move_to_algebraic(board, move, buffer, sizeof(buffer));
    printf("%s", buffer);
}

// Print a full game with positions at specified intervals
void print_game_debug(const Board *positions, int num_positions, int interval) {
    printf("\n=== Game Debug (showing every %d positions) ===\n", interval);

    for (int i = 0; i < num_positions; i += interval) {
        printf("\nPosition %d:\n", i);
        print_board_pretty(&positions[i]);

        // Also show evaluation
        float eval = evaluate_position(&positions[i]);
        printf("Classical evaluation: %.2f\n", eval);

#if HAVE_ONNXRUNTIME
        // If we have a loaded neural model, show that evaluation too
        if (is_neural_initialized()) {
            float neural_eval = evaluate_neural(&positions[i]);
            printf("Neural evaluation: %.2f\n", neural_eval);
        }
#endif
    }
}

// Add this helper function to compare board positions
bool boards_equal(const Board *board1, const Board *board2) {
    // Compare pieces
    for (int sq = 0; sq < 64; sq++) {
        if (board1->pieces[sq].type != board2->pieces[sq].type ||
            board1->pieces[sq].color != board2->pieces[sq].color) {
            return false;
        }
    }

    // Compare additional state that matters for move detection
    if (board1->side_to_move != board2->side_to_move) return false;
    if (board1->castle_rights != board2->castle_rights) return false;
    if (board1->en_passant_square != board2->en_passant_square) return false;

    return true;
}

// Completely revised print_game_with_evals function
void print_game_with_evals(const Board *positions, float *evaluations, int num_positions) {
    if (num_positions <= 1) {
        printf("No moves to display\n");
        return;
    }

    printf("\n=== Game with Evaluations ===\n\n");

    // Print initial position
    print_board_pretty(&positions[0]);
    printf("Initial evaluation: %.2f\n\n", evaluations[0]);

    // Simple output format - show all positions and evaluations
    for (int i = 1; i < num_positions; i++) {
        printf("Position %d (move %d):\n", i, (i + 1) / 2);

        // Determine side to move in previous position
        bool white_to_move = (positions[i - 1].side_to_move == WHITE);

        // Print move number
        if (white_to_move) {
            printf("%d. ", positions[i - 1].fullmove_number);
        } else {
            printf("%d... ", positions[i - 1].fullmove_number);
        }

        // Since we don't have the GamePosition structs here, we'll detect moves
        MoveList moves;
        generate_legal_moves(&positions[i - 1], &moves);
        bool found_move = false;

        for (int m = 0; m < moves.count; m++) {
            Board test_board = positions[i - 1];
            make_move(&test_board, &moves.moves[m]);

            if (boards_equal(&test_board, &positions[i])) {
                // Move found! Print it in algebraic notation
                char move_buffer[16];
                move_to_algebraic(&positions[i - 1], &moves.moves[m], move_buffer, sizeof(move_buffer));
                printf("%s (Eval: %.2f)\n", move_buffer, evaluations[i]);
                found_move = true;
                break;
            }
        }

        if (!found_move) {
            printf("[Move not detected] (Eval: %.2f)\n", evaluations[i]);
        }

        // Print board every 5 moves
        if (i % 10 == 0) {
            print_board_pretty(&positions[i]);
            printf("\n");
        }
    }

    // Final position
    printf("\nFinal position:\n");
    print_board_pretty(&positions[num_positions - 1]);
    printf("Final evaluation: %.2f\n", evaluations[num_positions - 1]);
}

// Add a simpler alternative function that just shows positions without move detection
void print_positions_with_evals(const Board *positions, float *evaluations, int num_positions) {
    printf("\n=== Game Positions ===\n\n");

    for (int i = 0; i < num_positions; i++) {
        printf("\nPosition %d (Fullmove: %d):\n", i, positions[i].fullmove_number);
        print_board_pretty(&positions[i]);
        printf("Evaluation: %.2f\n", evaluations[i]);

        if (i > 0) {
            // Show what changed from previous position
            printf("Changes from previous position:\n");
            bool changes_found = false;

            for (int sq = 0; sq < 64; sq++) {
                Piece prev_piece = positions[i - 1].pieces[sq];
                Piece curr_piece = positions[i].pieces[sq];

                // Only show actual changes (not empty → empty)
                if (prev_piece.type != curr_piece.type ||
                    prev_piece.color != curr_piece.color) {

                    // Skip if both are empty (this shouldn't happen, but let's be safe)
                    if (prev_piece.type == EMPTY && curr_piece.type == EMPTY) {
                        continue;
                    }

                    changes_found = true;
                    int file = SQUARE_FILE(sq);
                    int rank = SQUARE_RANK(sq);
                    printf("  Square %c%d: ", 'a' + file, rank + 1);

                    if (prev_piece.type != EMPTY) {
                        printf("%s → ", UNICODE_PIECES[prev_piece.type][prev_piece.color]);
                    } else {
                        printf("empty → ");
                    }

                    if (curr_piece.type != EMPTY) {
                        printf("%s\n", UNICODE_PIECES[curr_piece.type][curr_piece.color]);
                    } else {
                        printf("empty\n");
                    }
                }
            }

            if (!changes_found) {
                printf("  No piece changes detected!\n");
            }

            // Show other state changes
            const Board *prev = &positions[i - 1];
            const Board *curr = &positions[i];

            if (prev->side_to_move != curr->side_to_move) {
                printf("  Side to move: %s → %s\n",
                       prev->side_to_move == WHITE ? "White" : "Black",
                       curr->side_to_move == WHITE ? "White" : "Black");
            }

            if (prev->castle_rights != curr->castle_rights) {
                printf("  Castling rights changed\n");
            }

            if (prev->en_passant_square != curr->en_passant_square) {
                printf("  En passant square changed\n");
            }

            if (prev->fullmove_number != curr->fullmove_number) {
                printf("  Fullmove number: %d → %d\n",
                       prev->fullmove_number, curr->fullmove_number);

                // Detect negative or invalid fullmove numbers
                if (curr->fullmove_number < 1) {
                    printf("  ERROR: Invalid fullmove number detected: %d\n",
                           curr->fullmove_number);
                }
            }
        }
    }
}

// Check if neural evaluation is available
bool is_neural_initialized(void) {
#if HAVE_ONNXRUNTIME
    return get_neural_evaluator() != NULL;
#else
    return false;
#endif
}

// Add new function to print games with recorded moves

void print_game_with_recorded_moves(const GamePosition *game_positions, int num_positions) {
    if (num_positions <= 0) {
        printf("No positions to display\n");
        return;
    }

    printf("\n=== Game with Recorded Moves ===\n\n");

    // Print initial position
    print_board_pretty(&game_positions[0].board);
    printf("Initial evaluation: %.2f\n\n", game_positions[0].evaluation);

    // Track current line length for formatting
    int line_length = 0;
    const int max_line_length = 80;

    // Start from move 1 (positions 1 & 2)
    for (int move_num = 1; move_num <= (num_positions - 1) / 2; move_num++) {
        // Calculate position indices for white and black moves
        int white_pos = (move_num * 2) - 1;
        int black_pos = white_pos + 1;

        // Format: "N. white_move (eval) black_move (eval)  "
        char move_str[64] = {0};

        // Add move number
        int chars = sprintf(move_str, "%d. ", move_num);

        // Add White's move if available
        if (white_pos < num_positions && game_positions[white_pos].has_move) {
            char white_move[16];
            move_to_algebraic(&game_positions[white_pos - 1].board,
                              &game_positions[white_pos].last_move,
                              white_move, sizeof(white_move));
            chars += sprintf(move_str + chars, "%s (%.2f) ",
                             white_move, game_positions[white_pos].evaluation);
        } else {
            chars += sprintf(move_str + chars, "... ");
        }

        // Add Black's move if available
        if (black_pos < num_positions && game_positions[black_pos].has_move) {
            char black_move[16];
            move_to_algebraic(&game_positions[black_pos - 1].board,
                              &game_positions[black_pos].last_move,
                              black_move, sizeof(black_move));
            chars += sprintf(move_str + chars, "%s (%.2f)  ",
                             black_move, game_positions[black_pos].evaluation);
        }

        // Check if we need to start a new line
        if (line_length + chars > max_line_length) {
            printf("\n");
            line_length = 0;
        }

        // Print the move
        printf("%s", move_str);
        line_length += chars;

        // Print board at regular intervals (every 10 full moves)
        if (move_num % 10 == 0) {
            printf("\n\n");
            print_board_pretty(&game_positions[black_pos < num_positions ? black_pos : white_pos].board);
            printf("\n");
            line_length = 0;
        }
    }

    // Final position
    printf("\n\nFinal position:\n");
    print_board_pretty(&game_positions[num_positions - 1].board);
    printf("Final evaluation: %.2f\n", game_positions[num_positions - 1].evaluation);
}
