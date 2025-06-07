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
        printf(" %c ", 'a' + file);
    }
    printf("\n");

    // Print top border
    printf("   ┌");
    for (int file = 0; file < 8; file++) {
        printf("───");
        if (file < 7) printf("┬");
    }
    printf("┐\n");

    // Print board
    for (int rank = 7; rank >= 0; rank--) {
        printf(" %d │", rank + 1);

        for (int file = 0; file < 8; file++) {
            int square = SQUARE(file, rank);
            Piece piece = board->pieces[square];

            // Determine square color
            const char *bg_color = ((file + rank) % 2 == 0) ? ANSI_BG_GRAY : ANSI_BG_WHITE;
            const char *text_color = (piece.color == WHITE) ? ANSI_BRIGHT_WHITE : ANSI_BLACK;

            // Print piece with colors
            printf("%s%s %s %s", bg_color, text_color,
                   UNICODE_PIECES[piece.type][piece.color], ANSI_RESET);

            if (file < 7) printf("│");
        }

        printf("│ %d\n", rank + 1);

        // Print horizontal dividers
        if (rank > 0) {
            printf("   ├");
            for (int file = 0; file < 8; file++) {
                printf("───");
                if (file < 7) printf("┼");
            }
            printf("┤\n");
        }
    }

    // Print bottom border
    printf("   └");
    for (int file = 0; file < 8; file++) {
        printf("───");
        if (file < 7) printf("┴");
    }
    printf("┘\n");

    // Print bottom file labels
    printf("    ");
    for (int file = 0; file < 8; file++) {
        printf(" %c ", 'a' + file);
    }
    printf("\n");

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

// Convert a move to algebraic notation
char *move_to_algebraic(const Board *board, const Move *move, char *buffer, size_t buffer_size) {
    if (!move || !buffer || buffer_size < 6) return NULL;

    // Special case for castling
    if (move->castling) {
        if (SQUARE_FILE(move->to) > SQUARE_FILE(move->from)) {
            snprintf(buffer, buffer_size, "O-O");
        } else {
            snprintf(buffer, buffer_size, "O-O-O");
        }
        return buffer;
    }

    // Determine piece symbol
    char piece_symbol = ' ';
    int from_piece_type = board->pieces[move->from].type;

    switch (from_piece_type) {
    case KING:
        piece_symbol = 'K';
        break;
    case QUEEN:
        piece_symbol = 'Q';
        break;
    case ROOK:
        piece_symbol = 'R';
        break;
    case BISHOP:
        piece_symbol = 'B';
        break;
    case KNIGHT:
        piece_symbol = 'N';
        break;
    case PAWN:
        piece_symbol = ' ';
        break;
    default:
        piece_symbol = '?';
        break;
    }

    // Source and destination coordinates
    char src_file = 'a' + SQUARE_FILE(move->from);
    // Removed unused variable src_rank
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
            snprintf(buffer, buffer_size, "%cx%c%c", piece_symbol, dst_file, dst_rank);
        } else {
            snprintf(buffer, buffer_size, "%c%c%c", piece_symbol, dst_file, dst_rank);
        }
    }

    // Add promotion piece if applicable
    if (move->promotion != EMPTY) {
        char promotion_symbol = '?';
        switch (move->promotion) {
        case QUEEN:
            promotion_symbol = 'Q';
            break;
        case ROOK:
            promotion_symbol = 'R';
            break;
        case BISHOP:
            promotion_symbol = 'B';
            break;
        case KNIGHT:
            promotion_symbol = 'N';
            break;
        default:
            break;
        }

        size_t len = strlen(buffer);
        if (len + 2 < buffer_size) {
            buffer[len] = '=';
            buffer[len + 1] = promotion_symbol;
            buffer[len + 2] = '\0';
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

// Print a game with moves in algebraic notation and evaluations
void print_game_with_evals(const Board *positions, float *evaluations, int num_positions) {
    printf("\n=== Game with Evaluations ===\n");

    // Print initial position
    printf("\nInitial position:\n");
    print_board_pretty(&positions[0]);
    printf("Evaluation: %.2f\n", evaluations[0]);

    char move_buffer[8];

    // For each position, reconstruct the move that led to it
    for (int i = 1; i < num_positions; i++) {
        Board prev_board = positions[i - 1];
        Board curr_board = positions[i];

        // Determine the move made
        Move move = {0};
        bool found_move = false;

        MoveList moves;
        generate_legal_moves(&prev_board, &moves);

        for (int m = 0; m < moves.count; m++) {
            Board test_board = prev_board;
            make_move(&test_board, &moves.moves[m]);

            // Check if boards match
            bool match = true;
            for (int sq = 0; sq < 64; sq++) {
                if (test_board.pieces[sq].type != curr_board.pieces[sq].type ||
                    test_board.pieces[sq].color != curr_board.pieces[sq].color) {
                    match = false;
                    break;
                }
            }

            if (match) {
                move = moves.moves[m];
                found_move = true;
                break;
            }
        }

        // Print move information
        if (found_move) {
            printf("\nMove %d: ", (i + 1) / 2);
            if (prev_board.side_to_move == WHITE) {
                printf("%d. ", prev_board.fullmove_number);
            } else if (i == 1) {
                printf("%d... ", prev_board.fullmove_number);
            }

            move_to_algebraic(&prev_board, &move, move_buffer, sizeof(move_buffer));
            printf("%s", move_buffer);

            printf(" (Eval: %.2f)\n", evaluations[i]);

            // Every 10 full moves, also print the board
            if (i % 20 == 0) {
                print_board_pretty(&curr_board);
            }
        } else {
            printf("\nMove %d: [Unknown move] (Eval: %.2f)\n", (i + 1) / 2, evaluations[i]);
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
