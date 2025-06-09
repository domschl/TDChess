#include "zobrist.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// Random bitstrings for each piece on each square
// [piece_type][color][square]
static uint64_t piece_keys[7][2][64];  // +1 for EMPTY piece type

// Random bitstring for side to move (just one for BLACK, WHITE is 0)
static uint64_t side_key;

// Random bitstrings for en passant files (0-7)
static uint64_t en_passant_keys[8];

// Random bitstrings for castling rights (K, Q, k, q)
static uint64_t castling_keys[4];

// Helper function to generate a random 64-bit number
static uint64_t random_u64(void) {
    // We need a high-quality random number for each bit
    uint64_t r = 0;
    for (int i = 0; i < 64; i++) {
        r = (r << 1) | (rand() & 1);
    }
    return r;
}

// Initialize the Zobrist hashing system with random keys
void init_zobrist(void) {
    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Initialize piece keys
    for (int piece = 0; piece < 7; piece++) {
        for (int color = 0; color < 2; color++) {
            for (int square = 0; square < 64; square++) {
                piece_keys[piece][color][square] = random_u64();
            }
        }
    }

    // Initialize side to move key
    side_key = random_u64();

    // Initialize en passant keys (one for each file)
    for (int file = 0; file < 8; file++) {
        en_passant_keys[file] = random_u64();
    }

    // Initialize castling keys
    for (int i = 0; i < 4; i++) {
        castling_keys[i] = random_u64();
    }
}

// Compute a full Zobrist hash key for a board position
uint64_t compute_zobrist_key(const Board *board) {
    uint64_t hash = 0;

    // 1. Pieces on squares
    for (int square = 0; square < 64; square++) {
        Piece piece = board->pieces[square];
        if (piece.type != EMPTY) {
            hash ^= piece_keys[piece.type][piece.color][square];
        }
    }

    // 2. Side to move
    if (board->side_to_move == BLACK) {
        hash ^= side_key;
    }

    // 3. En passant square
    if (board->en_passant_square >= 0) {
        int file = board->en_passant_square % 8;
        hash ^= en_passant_keys[file];
    }

    // 4. Castling rights
    if (board->castle_rights & CASTLE_WHITE_KINGSIDE) {
        hash ^= castling_keys[0];
    }
    if (board->castle_rights & CASTLE_WHITE_QUEENSIDE) {
        hash ^= castling_keys[1];
    }
    if (board->castle_rights & CASTLE_BLACK_KINGSIDE) {
        hash ^= castling_keys[2];
    }
    if (board->castle_rights & CASTLE_BLACK_QUEENSIDE) {
        hash ^= castling_keys[3];
    }

    return hash;
}

// Update a Zobrist hash key incrementally when making a move
uint64_t update_zobrist_key(uint64_t key, const Board *board, const Move *move) {
    uint64_t new_key = key;

    // 1. Remove piece from source square
    PieceType piece_type = board->pieces[move->from].type;
    Color piece_color = board->pieces[move->from].color;
    new_key ^= piece_keys[piece_type][piece_color][move->from];

    // 2. If capture, remove captured piece from target square
    if (move->capture) {
        if (move->en_passant) {
            // For en passant, the captured pawn is on a different square
            int captured_pawn_square = move->captured_piece_square;
            new_key ^= piece_keys[PAWN][!piece_color][captured_pawn_square];
        } else {
            // Normal capture
            PieceType captured_type = board->pieces[move->to].type;
            Color captured_color = board->pieces[move->to].color;
            new_key ^= piece_keys[captured_type][captured_color][move->to];
        }
    }

    // 3. Add piece to target square (account for promotion)
    PieceType piece_to_add = (move->promotion != EMPTY) ? move->promotion : piece_type;
    new_key ^= piece_keys[piece_to_add][piece_color][move->to];

    // 4. Handle castling - rook moves
    if (piece_type == KING && abs(move->to - move->from) == 2) {
        // Kingside castling
        if (move->to > move->from) {
            int rook_from = (move->from | 7);  // H-file
            int rook_to = move->to - 1;        // F-file
            new_key ^= piece_keys[ROOK][piece_color][rook_from];
            new_key ^= piece_keys[ROOK][piece_color][rook_to];
        }
        // Queenside castling
        else {
            int rook_from = (move->from & ~7);  // A-file
            int rook_to = move->to + 1;         // D-file
            new_key ^= piece_keys[ROOK][piece_color][rook_from];
            new_key ^= piece_keys[ROOK][piece_color][rook_to];
        }
    }

    // 5. Flip side to move
    new_key ^= side_key;

    // 6. Clear old en passant if any
    if (board->en_passant_square >= 0) {
        int file = board->en_passant_square % 8;
        new_key ^= en_passant_keys[file];
    }

    // 7. Set new en passant if applicable (pawn double push)
    if (piece_type == PAWN && abs(move->to - move->from) == 16) {
        int file = move->from % 8;
        new_key ^= en_passant_keys[file];
    }

    // 8. Update castling rights if necessary
    int old_castling = board->castle_rights;
    int new_castling = old_castling;

    // King move removes all castling rights for that side
    if (piece_type == KING) {
        if (piece_color == WHITE) {
            new_castling &= ~(CASTLE_WHITE_KINGSIDE | CASTLE_WHITE_QUEENSIDE);
        } else {
            new_castling &= ~(CASTLE_BLACK_KINGSIDE | CASTLE_BLACK_QUEENSIDE);
        }
    }

    // Rook moves remove castling rights for that side
    if (piece_type == ROOK) {
        if (piece_color == WHITE) {
            if (move->from == 0) {  // a1
                new_castling &= ~CASTLE_WHITE_QUEENSIDE;
            } else if (move->from == 7) {  // h1
                new_castling &= ~CASTLE_WHITE_KINGSIDE;
            }
        } else {
            if (move->from == 56) {  // a8
                new_castling &= ~CASTLE_BLACK_QUEENSIDE;
            } else if (move->from == 63) {  // h8
                new_castling &= ~CASTLE_BLACK_KINGSIDE;
            }
        }
    }

    // Rook captures remove castling rights
    if (move->capture && !move->en_passant) {
        if (move->to == 0) {  // a1
            new_castling &= ~CASTLE_WHITE_QUEENSIDE;
        } else if (move->to == 7) {  // h1
            new_castling &= ~CASTLE_WHITE_KINGSIDE;
        } else if (move->to == 56) {  // a8
            new_castling &= ~CASTLE_BLACK_QUEENSIDE;
        } else if (move->to == 63) {  // h8
            new_castling &= ~CASTLE_BLACK_KINGSIDE;
        }
    }

    // Update hash with castling rights changes
    if (old_castling != new_castling) {
        // Remove old castling rights from hash
        if (old_castling & CASTLE_WHITE_KINGSIDE) {
            new_key ^= castling_keys[0];
        }
        if (old_castling & CASTLE_WHITE_QUEENSIDE) {
            new_key ^= castling_keys[1];
        }
        if (old_castling & CASTLE_BLACK_KINGSIDE) {
            new_key ^= castling_keys[2];
        }
        if (old_castling & CASTLE_BLACK_QUEENSIDE) {
            new_key ^= castling_keys[3];
        }

        // Add new castling rights to hash
        if (new_castling & CASTLE_WHITE_KINGSIDE) {
            new_key ^= castling_keys[0];
        }
        if (new_castling & CASTLE_WHITE_QUEENSIDE) {
            new_key ^= castling_keys[1];
        }
        if (new_castling & CASTLE_BLACK_KINGSIDE) {
            new_key ^= castling_keys[2];
        }
        if (new_castling & CASTLE_BLACK_QUEENSIDE) {
            new_key ^= castling_keys[3];
        }
    }

    return new_key;
}

// Get hash value for side to move
uint64_t side_to_move_hash(void) {
    return side_key;
}

// Get hash value for en passant square
uint64_t en_passant_hash(int square) {
    if (square < 0) {
        return 0;
    }
    return en_passant_keys[square % 8];
}

// Get hash value for castling rights
uint64_t castling_hash(int castling_rights) {
    uint64_t hash = 0;
    if (castling_rights & CASTLE_WHITE_KINGSIDE) {
        hash ^= castling_keys[0];
    }
    if (castling_rights & CASTLE_WHITE_QUEENSIDE) {
        hash ^= castling_keys[1];
    }
    if (castling_rights & CASTLE_BLACK_KINGSIDE) {
        hash ^= castling_keys[2];
    }
    if (castling_rights & CASTLE_BLACK_QUEENSIDE) {
        hash ^= castling_keys[3];
    }
    return hash;
}
