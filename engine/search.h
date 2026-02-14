#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include "movegen.h"
#include <stdint.h>
#include <stdbool.h>

// Search configuration
typedef struct {
    int max_depth;                 // Maximum search depth
    int time_limit_ms;             // Time limit in milliseconds, 0 for unlimited
    uint64_t tt_size;              // Transposition table size in entries
    bool use_null_move;            // Whether to use null move pruning
    bool use_iterative_deepening;  // Whether to use iterative deepening
    int verbosity;                 // 0=quiet, 1=normal, 2=verbose
} SearchConfig;

// Transposition table entry structure
typedef struct {
    uint64_t key;    // Zobrist hash key
    float score;     // Evaluation score
    uint8_t depth;   // Search depth
    uint8_t flag;    // Entry type (exact, alpha, beta)
    Move best_move;  // Best move from this position
} TTEntry;

// Default search configuration
extern const SearchConfig DEFAULT_SEARCH_CONFIG;

// Initialize search module with configuration
bool init_search(SearchConfig config);

// Clean up search module resources
void cleanup_search(void);

/**
 * @brief Find the best move in a position (compatibility wrapper for search_position)
 *
 * @param board The current chess position
 * @param depth Maximum search depth
 * @param best_move Pointer to store the best move found
 * @param nodes_searched Pointer to store the number of nodes searched
 * @param verbosity 0=quiet, 1=normal, 2=verbose
 * @return float Score of the position in pawn units
 */
float find_best_move(Board *board, int depth, Move *best_move, uint64_t *nodes_searched, int verbosity);

// Main search function - finds best move and returns evaluation
float search_position(Board *board, int depth, Move *best_move, uint64_t *nodes_searched);

// Alpha-beta search function
float alpha_beta(Board *board, int depth, float alpha, float beta, uint64_t *nodes,
                 int ply, uint64_t board_key, Move *pv_line, int *pv_length);

// Iterative deepening search
float iterative_deepening_search(Board *board, int max_depth, Move *best_move, uint64_t *nodes, int verbosity);

// Check if side has non-pawn material (for null move pruning)
bool has_non_pawn_material(const Board *board);

// Set the game history (hashes of previous positions) for repetition detection
void set_game_history(const uint64_t *hashes, int count);

#endif  // SEARCH_H
