#include "eval.h"
#include "movegen.h"
#include "visualization.h"
#include <stdio.h>

// Global evaluation type
static EvaluationType current_eval_type = EVAL_BASIC;

// Get current evaluation type
EvaluationType get_evaluation_type(void) {
    return current_eval_type;
}

// Set evaluation type
void set_evaluation_type(EvaluationType type) {
    current_eval_type = type;
}

// Main evaluation function that dispatches to the appropriate implementation
// This function will now ALWAYS return the evaluation from White's perspective, in pawn units.
float evaluate_position(const Board *board) {
    float score_pawn_units;

    switch (current_eval_type) {
    case EVAL_NEURAL:
        // evaluate_neural returns score in CENTIPAWNS from CURRENT PLAYER's perspective.
        float nn_score_current_player_centipawns = evaluate_neural(board);

        float nn_score_white_perspective_centipawns;
        if (board->side_to_move == WHITE) {
            // If White is to move, current player's score is already from White's perspective (relative to White)
            // but evaluate_neural gives it from White's actual view.
            nn_score_white_perspective_centipawns = nn_score_current_player_centipawns;
        } else {
            // If Black is to move, a positive score from current player (Black) is bad for White.
            // So, negate it to get White's perspective.
            nn_score_white_perspective_centipawns = -nn_score_current_player_centipawns;
        }
        // Convert to PAWN UNITS for evaluate_position's contract
        score_pawn_units = nn_score_white_perspective_centipawns / 100.0f;
        break;
    case EVAL_BASIC:
    default:
        // evaluate_basic returns score from White's perspective directly, in pawn units.
        score_pawn_units = evaluate_basic(board);
        break;
    }
    return score_pawn_units;
}

// Main quiescence check function
bool is_position_quiet(const Board *board) {
    switch (current_eval_type) {
    case EVAL_NEURAL:
        return is_quiet_neural(board);
    case EVAL_BASIC:
    default:
        return is_quiet_basic(board);
    }
}

// Implementation of basic quiescence checking
bool is_quiet_basic(const Board *board) {
    // Check if there are any captures available
    MoveList captures;
    generate_captures(board, &captures);

    // Check if in check
    int king_square = -1;
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type == KING && board->pieces[sq].color == board->side_to_move) {
            king_square = sq;
            break;
        }
    }

    bool in_check = (king_square != -1) &&
                    is_square_attacked(board, king_square, !board->side_to_move);

    // A position is quiet if there are no captures and not in check
    return captures.count == 0 && !in_check;
}

bool is_quiet_neural(const Board *board) {
    // For now, just call basic quiescence check
    return is_quiet_basic(board);
}

// Print detailed evaluation information for debugging
void print_evaluation_details(const Board *board) {
    printf("Evaluation details for position:\n");
    print_board_pretty(board);

    float score = evaluate_position(board);
    bool is_quiet = is_position_quiet(board);

    printf("Evaluation score: %.2f\n", score);
    printf("Position is %s\n", is_quiet ? "quiet" : "tactical");

    // Print material balance
    int white_material = 0;
    int black_material = 0;

    for (int sq = 0; sq < 64; sq++) {
        Piece piece = board->pieces[sq];
        if (piece.type == EMPTY) continue;

        int value = 0;
        switch (piece.type) {
        case PAWN:
            value = PAWN_VALUE;
            break;
        case KNIGHT:
            value = KNIGHT_VALUE;
            break;
        case BISHOP:
            value = BISHOP_VALUE;
            break;
        case ROOK:
            value = ROOK_VALUE;
            break;
        case QUEEN:
            value = QUEEN_VALUE;
            break;
        default:
            break;
        }

        if (piece.color == WHITE) {
            white_material += value;
        } else {
            black_material += value;
        }
    }

    printf("Material: White=%d, Black=%d, Diff=%d\n",
           white_material, black_material, white_material - black_material);

    // Check for pieces under attack
    printf("Pieces under attack:\n");
    for (int sq = 0; sq < 64; sq++) {
        Piece piece = board->pieces[sq];
        if (piece.type == EMPTY) continue;

        if (is_square_attacked(board, sq, !piece.color)) {
            printf("  %c at %c%d is under attack\n",
                   "PNBRQK"[piece.type - 1],
                   'a' + SQUARE_FILE(sq),
                   SQUARE_RANK(sq) + 1);
        }
    }
}
// Global evaluation function pointer
static EvaluationFunction current_eval_function = evaluate_basic;

// Piece-square tables for positional evaluation (simplified)
// Values are in centipawns and represent bonuses/penalties

// Pawn piece-square table (white perspective)
static const int pawn_pst[64] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0};

// Knight piece-square table
static const int knight_pst[64] = {
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50};

// Bishop piece-square table
static const int bishop_pst[64] = {
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 5, 5, 5, 5, -10,
    -10, 0, 5, 0, 0, 5, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20};

// Rook piece-square table
static const int rook_pst[64] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0};

// Queen piece-square table
static const int queen_pst[64] = {
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20};

// King piece-square table (middlegame)
static const int king_pst[64] = {
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20};

// Get piece-square table value, flipping for black
static int get_pst_value(int square, PieceType piece, Color color) {
    // Flip square for black perspective
    int pst_square = (color == WHITE) ? square : (63 - square);

    switch (piece) {
    case PAWN:
        return pawn_pst[pst_square];
    case KNIGHT:
        return knight_pst[pst_square];
    case BISHOP:
        return bishop_pst[pst_square];
    case ROOK:
        return rook_pst[pst_square];
    case QUEEN:
        return queen_pst[pst_square];
    case KING:
        return king_pst[pst_square];
    default:
        return 0;
    }
}

// Basic evaluation function that considers material and piece positioning
// This function will now return the evaluation from WHITE'S PERSPECTIVE, in pawn units.
float evaluate_basic(const Board *board) {
    int material_score_centipawns = 0;
    int positional_score_centipawns = 0;

    // Piece values in centipawns
    const int piece_values[7] = {0, 100, 320, 330, 500, 900, 20000};  // EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

    for (int sq = 0; sq < 64; sq++) {
        Piece piece = board->pieces[sq];
        if (piece.type != EMPTY) {
            int value_cp = piece_values[piece.type];
            // get_pst_value is assumed to return the PST value for the piece at its square,
            // adjusted for color (e.g., by flipping the square for black pieces and using white's tables),
            // such that a positive value is good for that piece from White's overall perspective.
            int pst_val_cp = get_pst_value(sq, piece.type, piece.color);

            if (piece.color == WHITE) {
                material_score_centipawns += value_cp;
                positional_score_centipawns += pst_val_cp;
            } else {                                        // Black piece
                material_score_centipawns -= value_cp;      // Subtract black's material value from white's score
                positional_score_centipawns -= pst_val_cp;  // Subtract black's positional value (which was from white's view)
            }
        }
    }

    int total_score_centipawns = material_score_centipawns + positional_score_centipawns;

    // Convert to pawn units
    float score_pawn_units = (float)total_score_centipawns / 100.0f;

    return score_pawn_units;  // This is now always from White's perspective
}

// Get current evaluation function
EvaluationFunction get_evaluation_function(void) {
    return current_eval_function;
}

// Set evaluation function
void set_evaluation_function(EvaluationFunction eval_func) {
    if (eval_func != NULL) {
        current_eval_function = eval_func;
    }
}
