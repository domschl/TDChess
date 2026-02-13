#include "search.h"
#include "movegen.h"
#include "eval.h"
#include "board.h"
#include "zobrist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>

// Maximum plies in search
#define MAX_PLY 64

// Maximum depth for quiescence search
#define MAX_QUIESCENCE_DEPTH 8

// Time limit in milliseconds for search
#define TIME_LIMIT_MS 0  // 0 means no time limit

// Transposition table entry types
#define TT_EXACT 0
#define TT_ALPHA 1
#define TT_BETA 2

// Default search configuration
const SearchConfig DEFAULT_SEARCH_CONFIG = {
    .max_depth = 4,
    .time_limit_ms = TIME_LIMIT_MS,
    .tt_size = 1024 * 1024,  // 1 million entries
    .use_null_move = true,
    .use_iterative_deepening = true,
    .verbosity = 1};

// Killer moves storage (non-capture moves that caused beta cutoffs)
static Move killer_moves[MAX_PLY][2];

// History heuristic table [piece_type][to_square]
static int history_table[7][64];

// Transposition table
static TTEntry *transposition_table = NULL;
static uint64_t tt_size = 0;
static uint64_t tt_mask = 0;

// Search configuration
static SearchConfig search_config;

// Time control variables
static clock_t search_start_time;
static bool search_time_up = false;

// Forward declarations for internal functions
static bool is_in_check(const Board *board, Color side);
static bool is_draw_by_repetition(const Board *board);
static float quiescence_search(Board *board, float alpha, float beta, uint64_t *nodes, int qdepth, int current_ply);
static void sort_moves(MoveList *moves, int start_idx);
static int get_piece_value(PieceType piece);
static void score_capture(Move *move, const Board *board, int *score);

// Check if time for search has elapsed
static bool is_time_up(void) {
    if (search_config.time_limit_ms == 0) {
        return false;  // No time limit
    }

    clock_t current = clock();
    double elapsed_ms = (double)(current - search_start_time) * 1000.0 / CLOCKS_PER_SEC;

    // Stop if we've used 95% of our allocated time
    if (elapsed_ms >= search_config.time_limit_ms * 0.95) {
        search_time_up = true;
        return true;
    }

    return false;
}

// Initialize search with configuration
bool init_search(SearchConfig config) {
    search_config = config;

    // Find the largest power of 2 <= tt_size
    uint64_t size_pow2 = 1;
    while (size_pow2 * 2 <= config.tt_size) {
        size_pow2 *= 2;
    }

    tt_size = size_pow2;
    tt_mask = tt_size - 1;  // Mask for fast modulo with power of 2

    // Free old table if it exists
    if (transposition_table != NULL) {
        free(transposition_table);
        transposition_table = NULL;
    }

    // Allocate transposition table
    transposition_table = calloc(tt_size, sizeof(TTEntry));
    if (!transposition_table) {
        fprintf(stderr, "Failed to allocate transposition table of size %" PRIu64 " entries\n", tt_size);
        return false;
    }

    printf("Allocated transposition table with %" PRIu64 " entries (%.1f MB)\n",
           tt_size, (float)(tt_size * sizeof(TTEntry)) / (1024 * 1024));

    // Clear killer moves and history table
    memset(killer_moves, 0, sizeof(killer_moves));
    memset(history_table, 0, sizeof(history_table));

    // Initialize Zobrist keys
    init_zobrist();

    return true;
}

// Clean up search data structures
void cleanup_search(void) {
    if (transposition_table != NULL) {
        free(transposition_table);
        transposition_table = NULL;
    }
}

// Store position in transposition table
void tt_store(uint64_t key, float score, int depth, int flag, Move best_move) {
    // Use power-of-2 size for fast indexing
    uint64_t index = key & tt_mask;
    TTEntry *entry = &transposition_table[index];

    // Replacement strategy:
    // 1. Always replace if new position is searched deeper
    // 2. Or if it's the same position
    // 3. Or if the existing entry is from a much older search
    if (entry->depth <= depth || entry->key == key) {
        entry->key = key;
        entry->score = score;
        entry->depth = depth;
        entry->flag = flag;
        entry->best_move = best_move;
    }
}

// Probe transposition table
bool tt_probe(uint64_t key, float *score, int depth, int alpha, int beta, Move *move) {
    if (transposition_table == NULL) {
        return false;
    }

    uint64_t index = key & tt_mask;
    TTEntry *entry = &transposition_table[index];

    if (entry->key == key) {
        *move = entry->best_move;

        // Only use the score if the depth is sufficient
        if (entry->depth >= depth) {
            *score = entry->score;

            // Based on the stored flag, we can return immediately in some cases
            if (entry->flag == TT_EXACT) {
                return true;
            }

            // If this is a lower bound and it causes a beta cutoff
            if (entry->flag == TT_BETA && entry->score >= beta) {
                *score = beta;
                return true;
            }

            // If this is an upper bound and it causes an alpha cutoff
            if (entry->flag == TT_ALPHA && entry->score <= alpha) {
                *score = alpha;
                return true;
            }
        }
    }

    return false;
}

// Update killer moves
void update_killer_move(Move move, int ply) {
    // Only store non-captures as killer moves
    if (!move.capture) {
        // Don't store the same move twice
        if (memcmp(&killer_moves[ply][0], &move, sizeof(Move)) != 0) {
            // Shift existing killer move
            killer_moves[ply][1] = killer_moves[ply][0];
            killer_moves[ply][0] = move;
        }
    }
}

// Update history heuristic
void update_history(Move move, int depth, const Board *board) {
    // Only update for non-captures
    if (!move.capture) {
        // Get the piece type from the board using the from square
        PieceType piece_type = board->pieces[move.from].type;

        // Exponential history scheme
        history_table[piece_type][move.to] += depth * depth;

        // Keep values in reasonable range
        if (history_table[piece_type][move.to] > 10000) {
            // Scale down all history values
            for (int p = 0; p < 7; p++) {
                for (int sq = 0; sq < 64; sq++) {
                    history_table[p][sq] /= 2;
                }
            }
        }
    }
}

// Sort moves based on scores
void sort_moves(MoveList *moves, int start_idx) {
    int best_idx = start_idx;
    int best_score = moves->scores[start_idx];

    // Find the move with the highest score
    for (int i = start_idx + 1; i < moves->count; i++) {
        if (moves->scores[i] > best_score) {
            best_idx = i;
            best_score = moves->scores[i];
        }
    }

    // Swap with the current position if needed
    if (best_idx != start_idx) {
        Move temp_move = moves->moves[start_idx];
        int temp_score = moves->scores[start_idx];

        moves->moves[start_idx] = moves->moves[best_idx];
        moves->scores[start_idx] = moves->scores[best_idx];

        moves->moves[best_idx] = temp_move;
        moves->scores[best_idx] = temp_score;
    }
}

// Enhanced move scoring for better ordering
void score_moves(Board *board, MoveList *moves, Move tt_move, int ply) {
    for (int i = 0; i < moves->count; i++) {
        Move *move = &moves->moves[i];
        int score = 0;

        // 1. Hash move (best by far)
        if (move->from == tt_move.from && move->to == tt_move.to) {
            score = 20000;
        }
        // 2. Captures with MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        else if (move->capture) {
            // MVV (Most Valuable Victim)
            int victim_value = 0;
            if (move->en_passant) {
                victim_value = 100;  // Pawn value for en passant
            } else {
                PieceType victim_type = board->pieces[move->to].type;
                switch (victim_type) {
                case PAWN:
                    victim_value = 100;
                    break;
                case KNIGHT:
                    victim_value = 320;
                    break;
                case BISHOP:
                    victim_value = 330;
                    break;
                case ROOK:
                    victim_value = 500;
                    break;
                case QUEEN:
                    victim_value = 900;
                    break;
                default:
                    break;
                }
            }

            // LVA (Least Valuable Attacker)
            int attacker_value = 0;
            PieceType attacker_type = board->pieces[move->from].type;
            switch (attacker_type) {
            case PAWN:
                attacker_value = 100;
                break;
            case KNIGHT:
                attacker_value = 320;
                break;
            case BISHOP:
                attacker_value = 330;
                break;
            case ROOK:
                attacker_value = 500;
                break;
            case QUEEN:
                attacker_value = 900;
                break;
            case KING:
                attacker_value = 1000;
                break;
            default:
                break;
            }

            // Score = 10000 + MVV*10 - LVA
            // This ensures all captures are tried before non-captures
            // Higher-value victims are tried first
            // For same victim, lower-value attackers are tried first
            score = 10000 + victim_value * 10 - attacker_value;
        }
        // 3. Killer moves
        else if (move->from == killer_moves[ply][0].from && move->to == killer_moves[ply][0].to) {
            score = 9000;
        } else if (move->from == killer_moves[ply][1].from && move->to == killer_moves[ply][1].to) {
            score = 8000;
        }
        // 4. History heuristic
        else {
            PieceType piece_type = board->pieces[move->from].type;
            score = history_table[piece_type][move->to];
        }

        // Promotions get extra bonus
        if (move->promotion != EMPTY) {
            int promotion_bonus = 0;
            switch (move->promotion) {
            case QUEEN:
                promotion_bonus = 800;
                break;
            case ROOK:
                promotion_bonus = 500;
                break;
            case BISHOP:
                promotion_bonus = 300;
                break;
            case KNIGHT:
                promotion_bonus = 300;
                break;
            default:
                break;
            }
            score += promotion_bonus;
        }

        moves->scores[i] = score;
    }
}

// Main search function
float search_position(Board *board, int depth, Move *best_move, uint64_t *nodes_searched) {
    // Ensure search is initialized
    if (transposition_table == NULL) {
        if (!init_search(DEFAULT_SEARCH_CONFIG)) {
            fprintf(stderr, "ERROR: Failed to initialize search\n");
            return 0.0f;
        }
    }

    *nodes_searched = 0;
    search_time_up = false;
    search_start_time = clock();

    // Use iterative deepening if enabled
    float score;
    if (search_config.use_iterative_deepening) {
        score = iterative_deepening_search(board, depth, best_move, nodes_searched, search_config.verbosity);
    } else {
        // Initialize PV for direct alpha-beta search
        Move pv_line[MAX_PLY];
        int pv_length = 0;

        // Compute board hash key
        uint64_t board_key = compute_zobrist_key(board);

        // Run alpha-beta search
        score = alpha_beta(board, depth, -FLT_MAX, FLT_MAX, nodes_searched, 0,
                           board_key, pv_line, &pv_length);

        // Copy best move from PV if available
        if (pv_length > 0) {
            *best_move = pv_line[0];
        } else {
            // Fallback: just pick the first legal move
            MoveList moves;
            generate_legal_moves(board, &moves);
            if (moves.count > 0) {
                *best_move = moves.moves[0];
            }
        }
    }

    // Debug: Print the final score and move
    if (search_config.verbosity > 0) {
        char move_str[10];
        strcpy(move_str, move_to_string(*best_move));
        printf("info string Final score: %.2f, Best move: %s\n", score, move_str);
    }

    return score;
}

// Iterative deepening search
float iterative_deepening_search(Board *board, int max_depth, Move *best_move, uint64_t *nodes, int verbosity) {
    *nodes = 0;
    float best_score = 0.0f;
    Move pv_line[MAX_PLY];
    int pv_length = 0;

    // Initialize a dummy best move in case search is stopped early
    MoveList moves;
    generate_legal_moves(board, &moves);
    if (moves.count > 0) {
        *best_move = moves.moves[0];
    } else {
        // No legal moves - checkmate or stalemate
        return is_in_check(board, board->side_to_move) ? -1000.0f : 0.0f;
    }

    // Calculate the board's Zobrist hash key
    uint64_t board_key = compute_zobrist_key(board);

    // Start time measurement
    double start_ms = (double)clock() * 1000.0 / CLOCKS_PER_SEC;

    // Iterative deepening loop
    for (int depth = 1; depth <= max_depth; depth++) {
        float alpha = -FLT_MAX;
        float beta = FLT_MAX;

        // Search with current depth
        float score = alpha_beta(board, depth, alpha, beta, nodes, 0, board_key, pv_line, &pv_length);

        // If we're out of time, use the last completed depth
        if (search_time_up) {
            if (verbosity > 0) {
                printf("info string Time limit reached, stopping at depth %d\n", depth - 1);
            }
            break;
        }

        // Update best move and score only if the search completed
        best_score = score;
        if (pv_length > 0) {
            *best_move = pv_line[0];
        }

        // Print current iteration info
        if (verbosity > 0) {
            double elapsed_ms = (double)clock() * 1000.0 / CLOCKS_PER_SEC - start_ms;

            // Convert score to centipawns with proper rounding
            int score_cp = (int)(score * 100.0f + (score >= 0 ? 0.5f : -0.5f));

            printf("info depth %d score cp %d nodes %" PRIu64 " time %.0f pv ",
                   depth, score_cp, *nodes, elapsed_ms);

            // Print PV
            Board temp_board = *board;
            for (int i = 0; i < pv_length; i++) {
                char move_str[10];
                strcpy(move_str, move_to_string(pv_line[i]));
                printf("%s ", move_str);

                make_move(&temp_board, &pv_line[i]);
            }
            printf("\n");
        }
    }

    if (verbosity > 0) {
        double elapsed_ms = (double)clock() * 1000.0 / CLOCKS_PER_SEC - start_ms;
        printf("info string Search completed in %.2f ms with final score: %.2f\n",
               elapsed_ms, best_score);
    }

    return best_score;
}

// Fix alpha-beta search with correct negamax implementation
float alpha_beta(Board *board, int depth, float alpha, float beta, uint64_t *nodes,
                 int ply, uint64_t board_key, Move *pv_line, int *pv_length) {
    // Initialize PV length
    *pv_length = 0;

    // Check for draw by repetition or fifty-move rule
    if (is_draw_by_repetition(board) || board->halfmove_clock >= 100) {
        return 0.0f;  // Draw
    }

    // Check if we're out of time
    if (((*nodes) & 1023) == 0 && is_time_up()) {
        return 0.0f;  // Return a neutral score when out of time
    }

    // Check transposition table
    Move tt_move = {0};
    float tt_score;
    if (tt_probe(board_key, &tt_score, depth, alpha, beta, &tt_move)) {
        return tt_score;
    }

    // Base case: leaf node or quiescence search
    if (depth <= 0) {
        return quiescence_search(board, alpha, beta, nodes, 0, ply);
    }

    (*nodes)++;

    // Generate and score all legal moves
    MoveList moves;
    generate_legal_moves(board, &moves);

    // Check for checkmate or stalemate
    if (moves.count == 0) {
        if (is_in_check(board, board->side_to_move)) {
            return -1000.0f + ply;  // Checkmate, adjusted for ply
        } else {
            return 0.0f;  // Stalemate
        }
    }

    // Score moves with enhanced ordering
    score_moves(board, &moves, tt_move, ply);

    int move_count = 0;
    float best_score = -FLT_MAX;  // Initialize to worst possible score
    Move best_move = {0};
    int tt_flag = TT_ALPHA;

    // Child PV storage
    Move child_pv[MAX_PLY];
    int child_pv_length = 0;

    // Iterate through all legal moves
    for (int i = 0; i < moves.count; i++) {
        sort_moves(&moves, i);
        Move current_move = moves.moves[i];
        move_count++;

        // Make the move on a copy of the board
        Board next_board = *board;
        make_move(&next_board, &current_move);

        // Update hash key incrementally
        uint64_t new_key = update_zobrist_key(board_key, board, &current_move);

        float score;

        // Principal Variation Search (PVS)
        if (move_count == 1) {
            // Full-window search for first move
            score = -alpha_beta(&next_board, depth - 1, -beta, -alpha, nodes,
                                ply + 1, new_key, child_pv, &child_pv_length);
        } else {
            // Reduced depth for later moves (late move reduction)
            int reduction = 0;
            if (depth >= 3 && move_count > 4 && !current_move.capture &&
                current_move.promotion == EMPTY && !is_in_check(board, board->side_to_move)) {
                reduction = 1;
            }

            // Null-window search with reduced depth
            score = -alpha_beta(&next_board, depth - 1 - reduction, -alpha - 0.01f, -alpha,
                                nodes, ply + 1, new_key, child_pv, &child_pv_length);

            // Re-search with full window if score might be better than alpha
            if (score > alpha && (reduction > 0 || score < beta)) {
                // If reduced, retry with full depth and proper bounds
                if (reduction > 0) {
                    score = -alpha_beta(&next_board, depth - 1, -beta, -alpha,
                                        nodes, ply + 1, new_key, child_pv, &child_pv_length);
                } else if (score < beta) {
                    // If within bounds but not reduced, do a full window search
                    score = -alpha_beta(&next_board, depth - 1, -beta, -alpha,
                                        nodes, ply + 1, new_key, child_pv, &child_pv_length);
                }
            }
        }

        // Update best score and move
        if (score > best_score) {
            best_score = score;
            best_move = current_move;

            // Update PV
            pv_line[0] = current_move;
            memcpy(pv_line + 1, child_pv, child_pv_length * sizeof(Move));
            *pv_length = child_pv_length + 1;

            // Update alpha if improvement
            if (score > alpha) {
                alpha = score;
                tt_flag = TT_EXACT;

                // Beta cutoff
                if (alpha >= beta) {
                    // Update killer moves and history
                    update_killer_move(current_move, ply);
                    update_history(current_move, depth, board);

                    tt_flag = TT_BETA;
                    best_score = beta;  // Return the beta bound
                    break;
                }
            }
        }
    }

    // Store position in transposition table
    tt_store(board_key, best_score, depth, tt_flag, best_move);

    return best_score;
}

// Ensure quiescence search is correctly evaluating positions
float quiescence_search(Board *board, float alpha, float beta, uint64_t *nodes, int qdepth, int current_ply) {
    // Check for maximum depth
    if (qdepth >= MAX_QUIESCENCE_DEPTH) {
        return evaluate_position(board);
    }

    // Check for draw by repetition or fifty-move rule
    if (is_draw_by_repetition(board) || board->halfmove_clock >= 100) {
        return 0.0f;  // Draw
    }

    (*nodes)++;

    // Stand pat score - ensure it's relative to side to move
    float raw_eval = evaluate_position(board);
    float stand_pat = (board->side_to_move == WHITE) ? raw_eval : -raw_eval;

    // Beta cutoff
    if (stand_pat >= beta) {
        return beta;
    }

    // Update alpha if improvement
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    // Generate and score only captures
    MoveList moves;
    generate_captures(board, &moves);

    // Score captures for better ordering
    for (int i = 0; i < moves.count; i++) {
        // Score moves by MVV-LVA
        score_capture(&moves.moves[i], board, &moves.scores[i]);
    }

    // Process each capture in order
    for (int i = 0; i < moves.count; i++) {
        // Sort remaining moves
        sort_moves(&moves, i);
        Move current_move = moves.moves[i];

        // Make the move on a copy of the board
        Board next_board = *board;
        make_move(&next_board, &current_move);

        // Skip if we're in check after the move
        if (is_in_check(&next_board, next_board.side_to_move)) {
            continue;
        }

        // Recursively search - ensure proper negation and score propagation
        float score = -quiescence_search(&next_board, -beta, -alpha, nodes, qdepth + 1, current_ply + 1);

        // Beta cutoff
        if (score >= beta) {
            return beta;
        }

        // Update alpha if improvement
        if (score > alpha) {
            alpha = score;
        }
    }

    return alpha;
}

/**
 * Score a capture move using MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
 *
 * @param move Pointer to the move to score
 * @param board Pointer to the current board state
 * @param score Pointer to store the calculated score
 */
static void score_capture(Move *move, const Board *board, int *score) {
    if (!move->capture) {
        *score = 0;
        return;
    }

    // Get the piece types
    PieceType victim_type = board->pieces[move->to].type;
    PieceType attacker_type = board->pieces[move->from].type;

    // MVV-LVA calculation
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
        victim_value = 900;
        break;  // Use constant when available
    default:
        break;
    }

    int attacker_value = 0;
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
        attacker_value = 900;
        break;  // Use constant when available
    case KING:
        attacker_value = 10000;
        break;  // King has highest value
    default:
        break;
    }

    // MVV-LVA formula: victim value * 10 - attacker value
    // This prioritizes capturing high-value pieces with low-value attackers
    *score = victim_value * 10 - attacker_value;

    // Add promotion bonus if applicable
    if (move->promotion != EMPTY) {
        switch (move->promotion) {
        case QUEEN:
            *score += 800;
            break;
        case ROOK:
            *score += 500;
            break;
        case BISHOP:
            *score += 300;
            break;
        case KNIGHT:
            *score += 300;
            break;
        default:
            break;
        }
    }
}

// Check if board is in check
bool is_in_check(const Board *board, Color side) {
    return is_square_attacked(board, board->king_pos[side], !side);
}

// Get piece value in centipawns
static int get_piece_value(PieceType piece) {
    switch (piece) {
    case PAWN:
        return 100;
    case KNIGHT:
        return 320;
    case BISHOP:
        return 330;
    case ROOK:
        return 500;
    case QUEEN:
        return 900;
    default:
        return 0;
    }
}

// Check if side has non-pawn material (for null move pruning)
bool has_non_pawn_material(const Board *board) {
    Color side = board->side_to_move;
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].color == side &&
            board->pieces[sq].type != EMPTY &&
            board->pieces[sq].type != PAWN &&
            board->pieces[sq].type != KING) {
            return true;
        }
    }
    return false;
}

// Simple draw by repetition check (basic implementation)
bool is_draw_by_repetition(const Board *board) {
    // This is a simple placeholder - a real implementation would track board history
    // For now, just rely on the 50-move rule
    return board->halfmove_clock >= 100;
}

/**
 * @brief Find the best move in a position (compatibility wrapper for search_position)
 *
 * This function is a wrapper around search_position to maintain compatibility
 * with existing code that uses the find_best_move interface.
 *
 * @param board The current chess position
 * @param depth Maximum search depth
 * @param best_move Pointer to store the best move found
 * @param nodes_searched Pointer to store the number of nodes searched
 * @param verbosity 0=quiet, 1=normal, 2=verbose
 * @return float Score of the position in pawn units
 */
float find_best_move(Board *board, int depth, Move *best_move, uint64_t *nodes_searched, int verbosity) {
    // Set verbosity level for this search
    int old_verbosity = search_config.verbosity;
    search_config.verbosity = verbosity;

    // Call the main search function
    float score = search_position(board, depth, best_move, nodes_searched);

    if (verbosity > 1) {
        printf("info string Raw score: %.4f, CP score: %d\n",
               score, (int)(score * 100.0f));
    }

    // Restore original verbosity
    search_config.verbosity = old_verbosity;

    // Debug printout to verify score
    if (verbosity > 0) {
        char move_str[10];
        strcpy(move_str, move_to_string(*best_move));
        printf("info string Best move: %s, Score: %.2f\n", move_str, score);
    }

    return score;
}
