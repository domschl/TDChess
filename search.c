#include "search.h"
#include "movegen.h"
#include "python_binding.h"
#include "eval.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <float.h> // For FLT_MAX

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

// Define these if not available from a shared header, ensure consistency (values in centipawns)
#define CHECKMATE_SCORE_Q (-10000.0f) // Score for the player being checkmated in Q-search
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
// DELTA_PRUNING_MARGIN should be defined (e.g., 200 centipawns)

// Quiescence search to evaluate tactical positions at the horizon
float quiescence_search(Board *board, float alpha, float beta, uint64_t *nodes, int qdepth) {
    (*nodes)++;

    // Safety break for excessive nodes
    if (*nodes >= MAX_QUIESCENCE_NODES) {
        float eval_white_view = evaluate_position(board);
        return (board->side_to_move == WHITE) ? eval_white_view : -eval_white_view;
    }

    // Determine if the current player is in check
    bool in_check = is_square_attacked(board, board->king_pos[board->side_to_move], !board->side_to_move);

    float stand_pat_current_player = 0.0f; // Will be set if !in_check

    if (!in_check) {
        // If not in check, evaluate the stand-pat score (current player's perspective)
        float eval_white_view = evaluate_position(board);
        stand_pat_current_player = (board->side_to_move == WHITE) ? eval_white_view : -eval_white_view;

        // Standard alpha-beta pruning based on stand-pat
        if (stand_pat_current_player >= beta) {
            return beta; // Fail-high, opponent won't allow this state
        }
        if (stand_pat_current_player > alpha) {
            alpha = stand_pat_current_player;
        }

        // If qdepth is exhausted and not in check, we rely on stand-pat (via alpha)
        if (qdepth <= 0) {
            return alpha;
        }
    }
    // If in_check:
    // - Stand-pat score is not used for alpha/beta updates here because a move is forced.
    // - qdepth <= 0 does not cause an immediate return; we must try to find an evasion.
    //   The qdepth will limit the depth of the evasion search recursively.

    MoveList moves;
    if (in_check) {
        generate_legal_moves(board, &moves); // Generate all legal moves to get out of check
    } else {
        generate_captures(board, &moves);    // Generate only captures if not in check
    }

    // Handle terminal nodes (checkmate or stalemate-like quiet positions)
    if (moves.count == 0) {
        if (in_check) {
            return CHECKMATE_SCORE_Q; // Checkmated (score for current player being mated)
        } else {
            // Not in check, and no captures were generated.
            // Alpha (possibly updated by stand_pat) is the best score.
            return alpha;
        }
    }

    // Move ordering: Score all generated moves.
    // This function should ideally prioritize good captures or effective check evasions.
    score_moves(board, &moves);

    for (int i = 0; i < moves.count; i++) {
        sort_moves(&moves, i); // Iteratively pick the best remaining move
        Move current_move = moves.moves[i];

        // Delta Pruning: Apply only if NOT in check and the move is a capture.
        if (!in_check && current_move.capture) {
            // Determine the value of the captured piece
            int victim_sq = current_move.to;
            if (current_move.en_passant) {
                // For en passant, the captured pawn is "behind" the 'to' square from the mover's perspective
                victim_sq = current_move.to + (board->side_to_move == WHITE ? -8 : 8);
            }
            Piece victim_piece_struct = board->pieces[victim_sq];

            if (victim_piece_struct.type != EMPTY) { // Should be true for valid captures
                const int piece_values_cp[7] = {0, PAWN_VALUE_CP, KNIGHT_VALUE_CP, BISHOP_VALUE_CP, ROOK_VALUE_CP, QUEEN_VALUE_CP, 0}; // King not capturable
                int captured_piece_value_cp = piece_values_cp[victim_piece_struct.type];

                // Convert values to pawn units for comparison with alpha (which is in pawn units)
                float captured_piece_pawn_units = (float)captured_piece_value_cp / 100.0f;
                float delta_margin_pawn_units = (float)DELTA_PRUNING_MARGIN / 100.0f; // DELTA_PRUNING_MARGIN is in centipawns

                // Use stand_pat_current_player calculated at the start of this function (when !in_check)
                if (stand_pat_current_player + captured_piece_pawn_units + delta_margin_pawn_units < alpha) {
                    continue; // Prune this move; it's unlikely to raise alpha
                }
            }
        }

        Board new_board = *board; // Create a copy to make the move on
        make_move(&new_board, &current_move);

        // Recursive call for the next state.
        // qdepth is decremented, limiting the search depth of tactical sequences/evasions.
        float score = -quiescence_search(&new_board, -beta, -alpha, nodes, qdepth - 1);

        // Standard Negamax updates
        if (score >= beta) {
            return beta; // Fail-high (cutoff)
        }
        if (score > alpha) {
            alpha = score;
        }
    }
    return alpha; // Return the best score found for the current player
}

// Alpha-beta search implementation
float alpha_beta(Board *board, int depth, float alpha, float beta, uint64_t *nodes) {
    // Increment node counter
    (*nodes)++;

    // Base case: leaf node
    if (depth <= 0) {
        // Check if position is quiet
        if (is_position_quiet(board)) {
            float eval_white_view = evaluate_position(board);
            // Convert to current player's perspective for Negamax
            return (board->side_to_move == WHITE) ? eval_white_view : -eval_white_view;
        } else {
            // Call quiescence search for tactical positions
            // quiescence_search will also return from its current player's perspective
            return quiescence_search(board, alpha, beta, nodes, MAX_QUIESCENCE_DEPTH);
        }
    }

    // Check for draw by repetition or 50-move rule
    if (board->halfmove_clock >= 100) {
        return 0.0f;  // Draw
    }

    // Generate legal moves
    MoveList moves;
    generate_legal_moves(board, &moves);

    // Check for terminal states
    if (moves.count == 0) {
        // Check if in check (checkmate)
        int king_square = -1;
        for (int sq = 0; sq < 64; sq++) {
            if (board->pieces[sq].type == KING && board->pieces[sq].color == board->side_to_move) {
                king_square = sq;
                break;
            }
        }

        if (king_square != -1 && is_square_attacked(board, king_square, !board->side_to_move)) {
            // Checkmate: return worst possible score
            return -1000.0f + (float)depth;  // Prefer checkmate sooner
        } else {
            // Stalemate: return draw score
            return 0.0f;
        }
    }

    // Simple move ordering for normal search
    // Prioritize captures and promotions
    for (int i = 0; i < moves.count; i++) {
        moves.scores[i] = 0;

        // Prioritize captures
        if (moves.moves[i].capture) {
            moves.scores[i] += 10000;

            // MVV-LVA scoring for captures
            int victim_square = moves.moves[i].to;
            // int attacker_square = moves.moves[i].from;

            PieceType victim_type = board->pieces[victim_square].type;
            // PieceType attacker_type = board->pieces[attacker_square].type;

            // Get values
            int victim_value = 0;
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

            moves.scores[i] += victim_value;
        }

        // Prioritize promotions
        if (moves.moves[i].promotion != EMPTY) {
            moves.scores[i] += 9000;
        }

        // Prioritize checks
        Board temp_board = *board;
        make_move(&temp_board, &(moves.moves[i]));

        // Find opponent's king
        int king_square = -1;
        for (int sq = 0; sq < 64; sq++) {
            if (temp_board.pieces[sq].type == KING && temp_board.pieces[sq].color == !board->side_to_move) {
                king_square = sq;
                break;
            }
        }

        if (king_square != -1 && is_square_attacked(&temp_board, king_square, board->side_to_move)) {
            moves.scores[i] += 5000;
        }
    }

    // Simple insertion sort for move ordering
    for (int i = 1; i < moves.count; i++) {
        Move temp_move = moves.moves[i];
        int temp_score = moves.scores[i];
        int j = i - 1;

        while (j >= 0 && moves.scores[j] < temp_score) {
            moves.moves[j + 1] = moves.moves[j];
            moves.scores[j + 1] = moves.scores[j];
            j--;
        }

        moves.moves[j + 1] = temp_move;
        moves.scores[j + 1] = temp_score;
    }

    float best_score = -FLT_MAX;

    // Try each move
    for (int i = 0; i < moves.count; i++) {
        // Make move
        Board new_board = *board;
        make_move(&new_board, &(moves.moves[i]));

        // Recursive search with negation (negamax)
        float score = -alpha_beta(&new_board, depth - 1, -beta, -alpha, nodes);

        // Update best score
        if (score > best_score) {
            best_score = score;
        }

        // Update alpha
        if (score > alpha) {
            alpha = score;
        }

        // Alpha-beta pruning
        if (alpha >= beta) {
            break;
        }
    }

    return best_score;
}

// Search function with time limit
SearchResult search_position(Board *board, int depth) {
    SearchResult result = {0};
    result.score = -FLT_MAX;
    result.nodes_searched = 0;
    result.depth_reached = 0;

    MoveList moves;
    generate_legal_moves(board, &moves);

    if (moves.count == 0) {
        // No legal moves
        return result;
    }

    // Default to first move
    result.best_move = moves.moves[0];

    // Search each move
    for (int i = 0; i < moves.count; i++) {
        Board new_board = *board;
        make_move(&new_board, &(moves.moves[i]));

        // Negate the score since we're evaluating from opponent's perspective
        float score = -alpha_beta(&new_board, depth - 1, -FLT_MAX, FLT_MAX, &result.nodes_searched);

        if (score > result.score) {
            result.score = score;
            result.best_move = moves.moves[i];
        }
    }

    result.depth_reached = depth;
    return result;
}

// Function to get a computer move
Move get_computer_move(Board *board, int depth) {
    printf("Computer is thinking (depth %d)...\n", depth);

    clock_t start = clock();
    SearchResult result = search_position(board, depth);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Best move: %s\n", move_to_string(result.best_move));
    printf("Score: %.2f\n", result.score);
    printf("Nodes searched: %llu\n", result.nodes_searched);
    printf("Time taken: %.2f seconds\n", time_taken);
    printf("Nodes per second: %.0f\n", result.nodes_searched / (time_taken > 0 ? time_taken : 1));

    return result.best_move;
}

// Update the find_best_move implementation to return the score

// Modified implementation to return the best score
float find_best_move(Board *board, int depth, Move *best_move, uint64_t *nodes) {
    // Reset node counter
    *nodes = 0;

    // Set initial alpha and beta values
    float alpha = -INFINITY; // Use INFINITY from float.h if available, or a very large number
    float beta = INFINITY;   // Use INFINITY from float.h

    // Generate all legal moves
    MoveList moves;
    generate_legal_moves(board, &moves);

    // If no legal moves, return score from current player's perspective
    if (moves.count == 0) {
        int king_square = -1;
        for (int sq = 0; sq < 64; sq++) {
            if (board->pieces[sq].type == KING && board->pieces[sq].color == board->side_to_move) {
                king_square = sq;
                break;
            }
        }
        if (king_square != -1 && is_square_attacked(board, king_square, !board->side_to_move)) {
            return -1000.0f;  // Checkmate (score for current player)
        } else {
            return 0.0f;  // Stalemate (score for current player)
        }
    }

    // Score moves for initial ordering
    score_moves(board, &moves); // Ensure score_moves is robust

    // Initialize best score and move
    float best_score_for_root_player = -INFINITY; // Or -FLT_MAX
    *best_move = moves.moves[0];  // Default to first move

    // Search each move
    for (int i = 0; i < moves.count; i++) {
        // Order moves by score (iteratively brings best to current 'i')
        sort_moves(&moves, i);

        Move current_move_to_try = moves.moves[i]; // Use a distinct variable for clarity
        make_move(board, &current_move_to_try);

        // Search from opponent's perspective
        float score_from_opponent = alpha_beta(board, depth - 1, -beta, -alpha, nodes);
        float score_for_current_player = -score_from_opponent;

        unmake_move(board, current_move_to_try);

        if (score_for_current_player > best_score_for_root_player) {
            best_score_for_root_player = score_for_current_player;
            *best_move = current_move_to_try;

            if (best_score_for_root_player > alpha) {
                alpha = best_score_for_root_player;
            }
        }
        // No beta check needed at the absolute root if we want to find the true best move
        // and not just one that's "good enough" for a previous search iteration.
        // If this were part of iterative deepening, a beta check (if alpha >= beta) break; would be here.
    }
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
