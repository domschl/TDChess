#ifndef SEARCH_H
#define SEARCH_H

#include "board.h"
#include "movegen.h"
#include "eval.h"

// Search result structure
typedef struct {
    Move best_move;
    float score;
    uint64_t nodes_searched;
    int depth_reached;
} SearchResult;

// Search function to find best move in a position
SearchResult search_position(Board *board, int depth);

// Alpha-beta search function
float alpha_beta(Board *board, int depth, float alpha, float beta, uint64_t *nodes);

// Function to make a computer move
Move get_computer_move(Board *board, int depth);

// Update the find_best_move declaration to return float instead of void
// Function to find the best move with a given depth
float find_best_move(Board *board, int depth, Move *best_move, uint64_t *nodes);

// Add these function declarations
void score_moves(Board *board, MoveList *moves);
void sort_moves(MoveList *moves, int start_index);

#endif  // SEARCH_H
