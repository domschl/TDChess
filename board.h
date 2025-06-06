#ifndef BOARD_H
#define BOARD_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

// Piece definitions
typedef enum {
    EMPTY = 0,
    PAWN = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK = 4,
    QUEEN = 5,
    KING = 6
} PieceType;

typedef enum {
    WHITE = 0,
    BLACK = 1
} Color;

typedef struct {
    PieceType type;
    Color color;
} Piece;

// Bitboard definition
typedef uint64_t Bitboard;

// Complete board representation
typedef struct {
    // Pieces on the board indexed by position (0-63)
    Piece pieces[64];

    // Bitboards for piece positions
    Bitboard piece_bb[2][7];  // [color][piece_type]

    // Combined bitboards
    Bitboard occupied[2];  // [color] - all pieces of a color
    Bitboard all_pieces;   // All pieces on the board

    // Game state
    Color side_to_move;
    int castle_rights;      // Bit flags for castling availability
    int en_passant_square;  // Square where en passant capture is possible (or -1)
    int halfmove_clock;     // For fifty-move rule
    int fullmove_number;    // Starting from 1, incremented after Black's move
} Board;

// Castling rights bits
#define CASTLE_WHITE_KINGSIDE 1
#define CASTLE_WHITE_QUEENSIDE 2
#define CASTLE_BLACK_KINGSIDE 4
#define CASTLE_BLACK_QUEENSIDE 8

// Board initialization
void init_board(Board *board);
void setup_default_position(Board *board);
bool parse_fen(Board *board, const char *fen);
bool board_to_fen(const Board *board, char *buffer, size_t buffer_size);

// Board utility functions
void print_board(const Board *board);
Piece get_piece(const Board *board, int square);
bool is_square_attacked(const Board *board, int square, Color by_side);
char get_piece_char(Piece piece);

// Square indices
#define A1 0
#define B1 1
#define C1 2
#define D1 3
#define E1 4
#define F1 5
#define G1 6
#define H1 7

#define A8 56
#define B8 57
#define C8 58
#define D8 59
#define E8 60
#define F8 61
#define G8 62
#define H8 63

// File and rank constants (renamed to avoid conflicts)
#define SQUARE_FILE_A 0
#define SQUARE_FILE_B 1
#define SQUARE_FILE_C 2
#define SQUARE_FILE_D 3
#define SQUARE_FILE_E 4
#define SQUARE_FILE_F 5
#define SQUARE_FILE_G 6
#define SQUARE_FILE_H 7

#define SQUARE_RANK_1 0
#define SQUARE_RANK_2 1
#define SQUARE_RANK_3 2
#define SQUARE_RANK_4 3
#define SQUARE_RANK_5 4
#define SQUARE_RANK_6 5
#define SQUARE_RANK_7 6
#define SQUARE_RANK_8 7

// Square macro to convert file and rank to square index
#define SQUARE(file, rank) ((rank) * 8 + (file))

// Macros to extract file and rank from square (renamed to avoid conflicts)
#define SQUARE_FILE(sq) ((sq) & 7)
#define SQUARE_RANK(sq) ((sq) >> 3)

// Bitboard operations
Bitboard square_to_bitboard(int square);
int bitboard_to_square(Bitboard bb);
int count_bits(Bitboard bb);
int pop_lsb(Bitboard *bb);
void validate_board_state(Board *board);

#endif  // BOARD_H
