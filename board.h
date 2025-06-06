#ifndef BOARD_H
#define BOARD_H

#include <stdint.h>
#include <stdbool.h>

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

// Move structure
typedef struct {
    int from;
    int to;
    PieceType promotion;
    bool capture;
    bool castling;
    bool en_passant;
    int captured_piece_square;  // For en passant
} Move;

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
char *board_to_fen(const Board *board);

// Board utility functions
void print_board(const Board *board);
Piece get_piece(const Board *board, int square);
bool is_square_attacked(const Board *board, int square, Color by_side);

// Square indexing macros
#define SQUARE(file, rank) ((rank) * 8 + (file))
#define RANK(square) ((square) / 8)
#define FILE(square) ((square) % 8)

// File and rank definitions
#define FILE_A 0
#define FILE_B 1
#define FILE_C 2
#define FILE_D 3
#define FILE_E 4
#define FILE_F 5
#define FILE_G 6
#define FILE_H 7

#define RANK_1 0
#define RANK_2 1
#define RANK_3 2
#define RANK_4 3
#define RANK_5 4
#define RANK_6 5
#define RANK_7 6
#define RANK_8 7

// Bitboard operations
Bitboard square_to_bitboard(int square);
int bitboard_to_square(Bitboard bb);
int count_bits(Bitboard bb);
int pop_lsb(Bitboard *bb);
void validate_board_state(Board *board);

#endif  // BOARD_H
