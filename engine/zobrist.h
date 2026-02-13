#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "board.h"
#include "movegen.h"
#include <stdint.h>
#include <stdbool.h>

// Initialize the Zobrist hashing system
void init_zobrist(void);

// Compute a full Zobrist hash key for a board position
uint64_t compute_zobrist_key(const Board *board);

// Update a Zobrist hash key incrementally when making a move
uint64_t update_zobrist_key(uint64_t key, const Board *board, const Move *move);

// Get hash value for side to move
uint64_t side_to_move_hash(void);

// Get hash value for en passant square
uint64_t en_passant_hash(int square);

// Get hash value for castling rights
uint64_t castling_hash(int castling_rights);

#endif  // ZOBRIST_H
