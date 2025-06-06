#include "search.h"
#include "movegen.h"
#include "python_binding.h"
#include "eval.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <float.h>

// Fix quiescence search to prevent endless loops and improve tactical evaluation

// Add a maximum quiescence depth constant
#define MAX_QUIESCENCE_DEPTH 4
#define MAX_QUIESCENCE_NODES 10000

// Check if time for search has elapsed
/*
static bool time_up(clock_t start_time, int time_limit_ms) {
    clock_t current = clock();
    double elapsed_ms = (double)(current - start_time) * 1000.0 / CLOCKS_PER_SEC;
    return elapsed_ms >= time_limit_ms;
}
*/

// Quiescence search to evaluate tactical positions at the horizon
float quiescence_search(Board *board, float alpha, float beta, uint64_t *nodes, int qdepth) {
    // Increment node counter
    (*nodes)++;

    // Safety check - limit total nodes in quiescence
    if (*nodes > MAX_QUIESCENCE_NODES) {
        return evaluate_position(board);
    }

    // Check if position is already quiet - if so, we can return the evaluation directly
    if (is_position_quiet(board)) {
        return evaluate_position(board);
    }

    // Get static evaluation first
    float stand_pat = evaluate_position(board);

    // Pruning opportunity - if we're already doing better than beta, opponent won't allow this position
    if (stand_pat >= beta) {
        return beta;
    }

    // Update alpha if static evaluation is better
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    // Hard depth limit to prevent excessive searching
    if (qdepth <= 0) {
        return stand_pat;
    }

    // Check if in check - if so, we need to generate all moves
    bool in_check = false;
    int king_square = -1;
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type == KING && board->pieces[sq].color == board->side_to_move) {
            king_square = sq;
            break;
        }
    }

    if (king_square != -1) {
        in_check = is_square_attacked(board, king_square, !board->side_to_move);
    }

    // Generate moves - either all legal moves if in check, or just captures
    MoveList moves;
    if (in_check) {
        generate_legal_moves(board, &moves);
    } else {
        generate_captures(board, &moves);
    }

    // No moves means either checkmate or stalemate if in check, otherwise just a quiet position
    if (moves.count == 0) {
        if (in_check) {
            return -1000.0f;  // Checkmate
        }
        return stand_pat;
    }

    // Simple move ordering - sort captures by MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
    // This is a basic implementation - just estimate the value of the capture
    for (int i = 0; i < moves.count; i++) {
        moves.scores[i] = 0;

        // If it's a capture, score by victim value - attacker value
        if (moves.moves[i].capture) {
            int victim_square = moves.moves[i].to;
            int attacker_square = moves.moves[i].from;

            PieceType victim_type = board->pieces[victim_square].type;
            PieceType attacker_type = board->pieces[attacker_square].type;

            // Get values
            int victim_value = 0;
            int attacker_value = 0;

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

            moves.scores[i] = victim_value - attacker_value / 10;
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

    // Try each move
    for (int i = 0; i < moves.count; i++) {
        // Skip non-captures if not in check and we're beyond the first level of quiescence
        if (!in_check && !moves.moves[i].capture && qdepth < MAX_QUIESCENCE_DEPTH) {
            continue;
        }

        Board new_board = *board;
        make_move(&new_board, &(moves.moves[i]));

        // Recursive search with negation (negamax)
        float score = -quiescence_search(&new_board, -beta, -alpha, nodes, qdepth - 1);

        // Update alpha
        if (score > alpha) {
            alpha = score;
        }

        // Alpha-beta pruning
        if (alpha >= beta) {
            return beta;
        }
    }

    return alpha;
}

// Alpha-beta search implementation
float alpha_beta(Board *board, int depth, float alpha, float beta, uint64_t *nodes) {
    // Increment node counter
    (*nodes)++;

    // Base case: leaf node
    if (depth <= 0) {
        // Check if position is quiet
        if (is_position_quiet(board)) {
            return evaluate_position(board);
        } else {
            // Call quiescence search for tactical positions
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

// Add the implementation
void find_best_move(Board *board, int depth, Move *best_move, uint64_t *nodes) {
    // Reset node counter
    *nodes = 0;

    // Set initial alpha and beta values
    float alpha = -INFINITY;
    float beta = INFINITY;

    // Generate all legal moves
    MoveList moves;
    generate_legal_moves(board, &moves);

    // If no legal moves, return
    if (moves.count == 0) {
        return;
    }

    // If only one legal move, return it immediately
    if (moves.count == 1) {
        *best_move = moves.moves[0];
        return;
    }

    // Score moves for initial ordering
    score_moves(board, &moves);

    // Initialize best score and move
    float best_score = -INFINITY;
    *best_move = moves.moves[0];  // Default to first move

    // Search each move
    for (int i = 0; i < moves.count; i++) {
        // Order moves by score
        sort_moves(&moves, i);

        // Make the move
        Move move = moves.moves[i];
        make_move(board, &move);

        // Search from opponent's perspective
        float score = -alpha_beta(board, depth - 1, -beta, -alpha, nodes);

        // Unmake the move - fixing the too few arguments error
        unmake_move(board, move);

        // Update best score and move if this move is better
        if (score > best_score) {
            best_score = score;
            *best_move = move;

            // Update alpha for alpha-beta pruning
            if (score > alpha) {
                alpha = score;
            }
        }
    }

    // Ensure we've set a best move - fixing the pointer syntax
    if (best_move->from == 0 && best_move->to == 0) {
        *best_move = moves.moves[0];  // Fall back to first move
    }
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
