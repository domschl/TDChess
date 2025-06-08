#include "search.h"
#include "movegen.h"
#include "eval.h"
#include "board.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// Forward declaration for quiescence_search
static float quiescence_search(Board *board, float alpha, float beta, uint64_t *nodes, int qdepth, int current_ply, Move *pv_line, int *pv_length);

// Fix quiescence search to prevent endless loops and improve tactical evaluation

// Add a maximum quiescence depth constant
#define MAX_QUIESCENCE_DEPTH 16
#define MAX_QUIESCENCE_NODES 10000
#define DELTA_PRUNING_MARGIN 200  // Value in centipawns (e.g., 200 for 2 pawns)

// Check if time for search has elapsed
/*
static bool time_up(clock_t start_time, int time_limit_ms) {
    clock_t current = clock();
    double elapsed_ms = (double)(current - start_time) * 1000.0 / CLOCKS_PER_SEC;
    return elapsed_ms >= time_limit_ms;
}
*/

// Define these if not available from a shared header, ensure consistency
// CHECKMATE_SCORE_Q_PAWN_UNITS is the score for the player being checkmated, in pawn units.
// A large negative number.
#define CHECKMATE_SCORE_Q_PAWN_UNITS (-1000.0f)  // e.g., -1000 pawns for being mated in q-search

// Piece values in CENTIPAWNS (ensure these are defined, e.g., in eval.h or config.h)
#ifndef PAWN_VALUE_CP
#define PAWN_VALUE_CP 100
#endif
#ifndef KNIGHT_VALUE_CP
#define KNIGHT_VALUE_CP 320
#endif
#ifndef BISHOP_VALUE_CP
#define BISHOP_VALUE_CP 330
#endif
#ifndef ROOK_VALUE_CP
#define ROOK_VALUE_CP 500
#endif
#ifndef QUEEN_VALUE_CP
#define QUEEN_VALUE_CP 900
#endif
// DELTA_PRUNING_MARGIN is in CENTIPAWNS (defined as 200 in search.c)

// Helper to get piece character for SAN (excluding pawns for non-captures)
static char get_san_piece_char(PieceType pt) {
    switch (pt) {
    case KNIGHT:
        return 'N';
    case BISHOP:
        return 'B';
    case ROOK:
        return 'R';
    case QUEEN:
        return 'Q';
    case KING:
        return 'K';
    case PAWN:
        return 'P';  // Used for pawn captures, e.g. Pxf3, but usually empty
    default:
        return ' ';
    }
}

// Convert a move to Standard Algebraic Notation (SAN)
// This is a simplified version, does not handle all disambiguation cases.
void move_to_san(const Board *board_before_move, const Move *move, char *san_buffer, size_t buffer_size) {
    if (!move || !san_buffer || buffer_size == 0) return;
    san_buffer[0] = '\0';

    if (move->castling) {
        if (move->to > move->from) {  // Kingside
            strncpy(san_buffer, "O-O", buffer_size - 1);
        } else {  // Queenside
            strncpy(san_buffer, "O-O-O", buffer_size - 1);
        }
    } else {
        Piece moving_piece = board_before_move->pieces[move->from];
        char piece_char_str[2] = {0};
        if (moving_piece.type != PAWN) {
            piece_char_str[0] = get_san_piece_char(moving_piece.type);
        }

        char from_sq_str[3];
        square_to_algebraic(move->from, from_sq_str);
        char to_sq_str[3];
        square_to_algebraic(move->to, to_sq_str);

        char capture_char_str[2] = {0};
        if (move->capture) {
            capture_char_str[0] = 'x';
            if (moving_piece.type == PAWN) {         // Pawn captures include from_file
                piece_char_str[0] = from_sq_str[0];  // e.g. 'e' from "exd5"
            }
        }

        char promotion_str[3] = {0};
        if (move->promotion != EMPTY) {
            promotion_str[0] = '=';
            promotion_str[1] = get_san_piece_char(move->promotion);
        }

        snprintf(san_buffer, buffer_size, "%s%s%s%s",
                 piece_char_str,
                 capture_char_str,  // If pawn capture, piece_char_str is from_file, so this 'x' is fine
                 to_sq_str,
                 promotion_str);
    }

    // Append check/checkmate (requires making the move on a temp board)
    Board after_move_board = *board_before_move;
    Move temp_move_for_check_eval = *move;                    // make_move might modify the move struct if it saves more state
    make_move(&after_move_board, &temp_move_for_check_eval);  // make_move needs a non-const Move*

    MoveList legal_moves_after;
    generate_legal_moves(&after_move_board, &legal_moves_after);

    Color opponent_color = (board_before_move->side_to_move == WHITE) ? BLACK : WHITE;
    if (is_square_attacked(&after_move_board, after_move_board.king_pos[opponent_color], board_before_move->side_to_move)) {
        if (legal_moves_after.count == 0) {  // Checkmate
            strncat(san_buffer, "#", buffer_size - strlen(san_buffer) - 1);
        } else {  // Check
            strncat(san_buffer, "+", buffer_size - strlen(san_buffer) - 1);
        }
    }
    san_buffer[buffer_size - 1] = '\0';  // Ensure null termination
}

// Alpha-beta search function
float alpha_beta(Board *board, int depth, float alpha, float beta, uint64_t *nodes, int current_ply, Move *pv_line, int *pv_length) {
    (*nodes)++;
    *pv_length = 0;  // Initialize PV length for this node

    // Base case: leaf node or quiescence search
    if (depth <= 0) {
        // return quiescence_search(board, alpha, beta, nodes, MAX_QUIESCENCE_DEPTH);
        // Quiescence search also needs PV parameters, though it typically doesn't extend PV
        Move q_pv_line[MAX_PLY];  // Local PV for qsearch, usually unused
        int q_pv_length = 0;
        return quiescence_search(board, alpha, beta, nodes, MAX_QUIESCENCE_DEPTH, current_ply, q_pv_line, &q_pv_length);  // Now this call is fine
    }

    // ... (existing fifty-move rule, repetition check logic) ...

    MoveList moves;
    generate_legal_moves(board, &moves);

    if (moves.count == 0) {
        if (is_square_attacked(board, board->king_pos[board->side_to_move], !board->side_to_move)) {
            return -1000.0f + (float)current_ply;  // Checkmate (score relative to current player, adjust for ply)
        } else {
            return 0.0f;  // Stalemate
        }
    }

    score_moves(board, &moves);  // For move ordering

    Move best_move_this_node = moves.moves[0];  // Default
    Move child_pv_line[MAX_PLY];
    int child_pv_length = 0;

    for (int i = 0; i < moves.count; i++) {
        sort_moves(&moves, i);
        Move current_move = moves.moves[i];
        make_move(board, &current_move);

        float score = -alpha_beta(board, depth - 1, -beta, -alpha, nodes, current_ply + 1, child_pv_line, &child_pv_length);

        unmake_move(board, current_move);

        if (score >= beta) {
            return beta;  // Fail high (cutoff)
        }
        if (score > alpha) {
            alpha = score;
            best_move_this_node = current_move;  // Store best move at this node

            // Construct PV for this node
            pv_line[0] = best_move_this_node;
            memcpy(&pv_line[1], child_pv_line, child_pv_length * sizeof(Move));
            *pv_length = 1 + child_pv_length;
        }
    }
    return alpha;
}

// Quiescence search
// pv_line is unused here but kept for signature consistency with alpha_beta
static float quiescence_search(Board *board, float alpha, float beta, uint64_t *nodes, int qdepth, int current_ply, Move *pv_line, int *pv_length) {
    pv_line;  // Mark pv_line as unused to suppress the warning

    (*nodes)++;
    *pv_length = 0;  // Quiescence search does not extend the PV from the main search

    // --- REVISED QSEARCH (ensure pv_length is set on all returns) ---
    bool in_check = is_square_attacked(board, board->king_pos[board->side_to_move], !board->side_to_move);
    float stand_pat_current_player_pawn_units = 0.0f;

    if (!in_check) {
        float eval_white_view_pawn_units = evaluate_position(board);
        stand_pat_current_player_pawn_units = (board->side_to_move == WHITE) ? eval_white_view_pawn_units : -eval_white_view_pawn_units;

        if (stand_pat_current_player_pawn_units >= beta) {
            *pv_length = 0;
            return beta;
        }
        if (stand_pat_current_player_pawn_units > alpha) {
            alpha = stand_pat_current_player_pawn_units;
        }
        if (qdepth <= 0) {
            *pv_length = 0;
            return alpha;
        }
    }

    MoveList moves;
    if (in_check) {
        generate_legal_moves(board, &moves);
    } else {
        generate_captures(board, &moves);
    }

    if (moves.count == 0) {
        *pv_length = 0;
        return in_check ? CHECKMATE_SCORE_Q_PAWN_UNITS : alpha;
    }

    score_moves(board, &moves);
    Move child_q_pv[1];  // Dummy, not used
    int child_q_pv_len = 0;

    for (int i = 0; i < moves.count; i++) {
        sort_moves(&moves, i);
        Move current_move = moves.moves[i];
        // ... (Delta Pruning logic as before) ...
        if (!in_check && current_move.capture) {
            // ... delta pruning logic ...
            // if (stand_pat_current_player_pawn_units + captured_piece_pawn_units + delta_margin_pawn_units < alpha) {
            //    continue;
            // }
        }

        Board new_board = *board;
        make_move(&new_board, &current_move);
        float score = -quiescence_search(&new_board, -beta, -alpha, nodes, qdepth - 1, current_ply + 1, child_q_pv, &child_q_pv_len);

        if (score >= beta) {
            *pv_length = 0;
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }
    *pv_length = 0;
    return alpha;
}

// Find the best move for the current position
float find_best_move(Board *board, int depth, Move *best_move, uint64_t *nodes) {
    *nodes = 0;
    float alpha = -FLT_MAX;  // Use FLT_MAX for float
    float beta = FLT_MAX;

    MoveList moves;
    generate_legal_moves(board, &moves);

    if (moves.count == 0) {
        // ... (checkmate/stalemate score as before) ...
        return is_square_attacked(board, board->king_pos[board->side_to_move], !board->side_to_move) ? -1000.0f : 0.0f;
    }

    score_moves(board, &moves);
    float best_score_for_root_player = -FLT_MAX;
    *best_move = moves.moves[0];  // Default

    Move overall_best_pv[MAX_PLY];
    int overall_best_pv_length = 0;
    Board original_board_state_at_root = *board;  // For SAN generation

    printf("info string Starting search at depth %d\n", depth);

    for (int i = 0; i < moves.count; i++) {
        sort_moves(&moves, i);
        Move current_move_to_try = moves.moves[i];
        Board temp_board = *board;  // Use a copy for each root move search
        make_move(&temp_board, &current_move_to_try);

        Move local_pv_line[MAX_PLY];
        int local_pv_length = 0;
        float score_from_opponent = alpha_beta(&temp_board, depth - 1, -beta, -alpha, nodes, 1, local_pv_line, &local_pv_length);
        float score_for_current_player = -score_from_opponent;

        // No unmake_move here as we used temp_board for the recursive call.
        // The original 'board' remains unchanged for the next root move.

        char root_move_san[10];
        move_to_san(&original_board_state_at_root, &current_move_to_try, root_move_san, sizeof(root_move_san));
        printf("info currmove %s (eval for this move: %.2f)\n", root_move_san, score_for_current_player * 100.0f);

        if (score_for_current_player > best_score_for_root_player) {
            best_score_for_root_player = score_for_current_player;
            *best_move = current_move_to_try;

            // Store the PV
            overall_best_pv[0] = current_move_to_try;
            memcpy(&overall_best_pv[1], local_pv_line, local_pv_length * sizeof(Move));
            overall_best_pv_length = 1 + local_pv_length;

            // Print PV (UCI-like format)
            // Score is in pawn units from current player's perspective. Convert to centipawns.
            // If side_to_move is Black, a positive score_for_current_player is good for Black.
            // UCI score is always from White's perspective.
            float uci_score_cp;
            if (original_board_state_at_root.side_to_move == WHITE) {
                uci_score_cp = best_score_for_root_player * 100.0f;
            } else {
                uci_score_cp = -best_score_for_root_player * 100.0f;  // Negate if Black's turn for White's perspective
            }

            printf("info depth %d score cp %.0f nodes %llu pv ", depth, uci_score_cp, *nodes);
            Board temp_board_for_san = original_board_state_at_root;
            for (int k = 0; k < overall_best_pv_length; k++) {
                char san_buffer[10];
                // Ensure the move struct passed to move_to_san is complete if make_move modifies it.
                // Here, overall_best_pv[k] should be the original move struct.
                move_to_san(&temp_board_for_san, &overall_best_pv[k], san_buffer, sizeof(san_buffer));
                printf("%s ", san_buffer);

                // make_move needs a non-const Move*. If overall_best_pv[k] is const, make a copy.
                Move temp_pv_move = overall_best_pv[k];
                make_move(&temp_board_for_san, &temp_pv_move);
            }
            printf("\n");

            if (best_score_for_root_player > alpha) {
                alpha = best_score_for_root_player;
            }
        }
        // No beta check at the absolute root for finding the true best move.
        // if (alpha >= beta) break; // This would be for fail-soft on root
    }
    printf("info string Search complete. Best score: %.2f cp\n", best_score_for_root_player * 100.0f);
    return best_score_for_root_player;
}

// Add implementations of the missing functions

// Score moves for ordering
void score_moves(Board *board, MoveList *moves) {
    for (int i = 0; i < moves->count; i++) {
        moves->scores[i] = 0;

        // Prioritize captures
        if (moves->moves[i].capture) {
            moves->scores[i] += 10000;

            // MVV-LVA scoring for captures
            int victim_square = moves->moves[i].to;
            int attacker_square = moves->moves[i].from;

            PieceType victim_type = board->pieces[victim_square].type;
            // We'll use attacker_type to avoid the unused variable warning
            PieceType attacker_type = board->pieces[attacker_square].type;

            // Get values
            int victim_value = 0;
            int attacker_value = 0;  // Use this to avoid the warning

            switch (victim_type) {
            case PAWN:
                victim_value = PAWN_VALUE;
                break;
            case KNIGHT:
                victim_value = KNIGHT_VALUE;
                break;
            case BISHOP:
                victim_value = BISHOP_VALUE;
                break;
            case ROOK:
                victim_value = ROOK_VALUE;
                break;
            case QUEEN:
                victim_value = QUEEN_VALUE;
                break;
            default:
                break;
            }

            switch (attacker_type) {
            case PAWN:
                attacker_value = PAWN_VALUE;
                break;
            case KNIGHT:
                attacker_value = KNIGHT_VALUE;
                break;
            case BISHOP:
                attacker_value = BISHOP_VALUE;
                break;
            case ROOK:
                attacker_value = ROOK_VALUE;
                break;
            case QUEEN:
                attacker_value = QUEEN_VALUE;
                break;
            default:
                break;
            }

            moves->scores[i] += victim_value - (attacker_value / 10);
        }

        // Prioritize promotions
        if (moves->moves[i].promotion != EMPTY) {
            moves->scores[i] += 9000;
        }
    }
}

// Sort moves starting from a given index
void sort_moves(MoveList *moves, int start_index) {
    // Find the best move from start_index to the end
    int best_index = start_index;
    for (int i = start_index + 1; i < moves->count; i++) {
        if (moves->scores[i] > moves->scores[best_index]) {
            best_index = i;
        }
    }

    // Swap the best move to the start_index position
    if (best_index != start_index) {
        Move temp_move = moves->moves[start_index];
        int temp_score = moves->scores[start_index];

        moves->moves[start_index] = moves->moves[best_index];
        moves->scores[start_index] = moves->scores[best_index];

        moves->moves[best_index] = temp_move;
        moves->scores[best_index] = temp_score;
    }
}
/* duplicate code, see eval.c
// Add this function to determine if a position is quiet (no captures possible)
bool is_position_quiet(const Board* board) {
    // A position is considered quiet if there are no captures available
    MoveList captures;
    generate_captures(board, &captures);

    // Also check if in check
    int king_square = -1;
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type == KING && board->pieces[sq].color == board->side_to_move) {
            king_square = sq;
            break;
        }
    }

    bool in_check = (king_square != -1) &&
                    is_square_attacked(board, king_square, !board->side_to_move);

    return captures.count == 0 && !in_check;
} */
