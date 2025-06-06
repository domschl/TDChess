#include "board.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

// Initialize the board to an empty state
void init_board(Board *board) {
    memset(board, 0, sizeof(Board));
    board->en_passant_square = -1;
    board->fullmove_number = 1;
}

// Set up the standard chess starting position
void setup_default_position(Board *board) {
    // Start with an empty board
    init_board(board);

    // Standard chess starting position in FEN
    const char *start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // Parse the FEN string to set up the board
    if (!parse_fen(board, start_fen)) {
        fprintf(stderr, "Error: Failed to set up default position\n");

        // Fall back to minimal position with just kings
        init_board(board);

        // Place kings in their starting positions
        board->pieces[SQUARE(FILE_E, RANK_1)].type = KING;
        board->pieces[SQUARE(FILE_E, RANK_1)].color = WHITE;
        board->pieces[SQUARE(FILE_E, RANK_8)].type = KING;
        board->pieces[SQUARE(FILE_E, RANK_8)].color = BLACK;

        // Update bitboards
        board->piece_bb[WHITE][KING] = square_to_bitboard(SQUARE(FILE_E, RANK_1));
        board->piece_bb[BLACK][KING] = square_to_bitboard(SQUARE(FILE_E, RANK_8));
        board->occupied[WHITE] = board->piece_bb[WHITE][KING];
        board->occupied[BLACK] = board->piece_bb[BLACK][KING];
        board->all_pieces = board->occupied[WHITE] | board->occupied[BLACK];
    }

    // Double check that bitboards are consistent with piece positions
    validate_board_state(board);
}

// Convert a FEN string to a board state
bool parse_fen(Board *board, const char *fen) {
    // Initialize the board to empty
    init_board(board);

    int rank = 7;   // Start from the 8th rank (index 7)
    int file = 0;   // Start from a-file (index 0)
    int field = 0;  // Track which part of the FEN we're parsing

    const char *ptr = fen;

    // Field 0: Piece placement
    while (*ptr && field == 0) {
        if (*ptr == '/') {
            rank--;
            file = 0;
        } else if (*ptr >= '1' && *ptr <= '8') {
            // Skip empty squares
            file += (*ptr - '0');
        } else if (strchr("PNBRQKpnbrqk", *ptr)) {
            // Place a piece
            if (file >= 8 || rank < 0 || rank >= 8) {
                fprintf(stderr, "FEN error: Invalid piece position at %c\n", *ptr);
                return false;
            }

            Piece piece;
            piece.color = islower(*ptr) ? BLACK : WHITE;
            char pc = tolower(*ptr);

            switch (pc) {
            case 'p':
                piece.type = PAWN;
                break;
            case 'n':
                piece.type = KNIGHT;
                break;
            case 'b':
                piece.type = BISHOP;
                break;
            case 'r':
                piece.type = ROOK;
                break;
            case 'q':
                piece.type = QUEEN;
                break;
            case 'k':
                piece.type = KING;
                break;
            default:
                fprintf(stderr, "FEN error: Invalid piece type %c\n", *ptr);
                return false;
            }

            int square = SQUARE(file, rank);
            board->pieces[square] = piece;

            // Update bitboards
            Bitboard sq_bb = square_to_bitboard(square);
            board->piece_bb[piece.color][piece.type] |= sq_bb;
            board->occupied[piece.color] |= sq_bb;
            board->all_pieces |= sq_bb;

            file++;
        } else if (*ptr == ' ') {
            // Move to next field
            field++;
        } else {
            fprintf(stderr, "FEN error: Invalid character %c\n", *ptr);
            return false;
        }

        ptr++;
    }

    // Field 1: Side to move
    if (!*ptr) return false;
    if (*ptr == 'w') {
        board->side_to_move = WHITE;
    } else if (*ptr == 'b') {
        board->side_to_move = BLACK;
    } else {
        fprintf(stderr, "FEN error: Expected 'w' or 'b' for side to move, got %c\n", *ptr);
        return false;
    }

    ptr++;
    if (!*ptr || *ptr != ' ') {
        fprintf(stderr, "FEN error: Expected space after side to move\n");
        return false;
    }
    ptr++;  // Skip space

    // Field 2: Castling rights
    field = 2;
    if (!*ptr) return false;

    if (*ptr == '-') {
        ptr++;  // No castling rights
    } else {
        while (*ptr && *ptr != ' ') {
            switch (*ptr) {
            case 'K':
                board->castle_rights |= CASTLE_WHITE_KINGSIDE;
                break;
            case 'Q':
                board->castle_rights |= CASTLE_WHITE_QUEENSIDE;
                break;
            case 'k':
                board->castle_rights |= CASTLE_BLACK_KINGSIDE;
                break;
            case 'q':
                board->castle_rights |= CASTLE_BLACK_QUEENSIDE;
                break;
            default:
                fprintf(stderr, "FEN error: Invalid castling right %c\n", *ptr);
                return false;
            }
            ptr++;
        }
    }

    if (!*ptr || *ptr != ' ') {
        fprintf(stderr, "FEN error: Expected space after castling rights\n");
        return false;
    }
    ptr++;  // Skip space

    // Field 3: En passant square
    field = 3;
    if (!*ptr) return false;

    if (*ptr == '-') {
        board->en_passant_square = -1;
        ptr++;
    } else if (*ptr >= 'a' && *ptr <= 'h') {
        int ep_file = *ptr - 'a';
        ptr++;

        if (!*ptr || *ptr < '1' || *ptr > '8') {
            fprintf(stderr, "FEN error: Invalid en passant rank\n");
            return false;
        }

        int ep_rank = *ptr - '1';
        board->en_passant_square = SQUARE(ep_file, ep_rank);
        ptr++;
    } else {
        fprintf(stderr, "FEN error: Invalid en passant square\n");
        return false;
    }

    if (!*ptr || *ptr != ' ') {
        fprintf(stderr, "FEN error: Expected space after en passant square\n");
        return false;
    }
    ptr++;  // Skip space

    // Field 4: Halfmove clock
    field = 4;
    if (!*ptr || !isdigit(*ptr)) {
        fprintf(stderr, "FEN error: Expected digit for halfmove clock\n");
        return false;
    }

    char halfmove_str[4] = {0};
    int i = 0;
    while (*ptr && isdigit(*ptr) && i < 3) {
        halfmove_str[i++] = *ptr++;
    }
    board->halfmove_clock = atoi(halfmove_str);

    if (!*ptr || *ptr != ' ') {
        fprintf(stderr, "FEN error: Expected space after halfmove clock\n");
        return false;
    }
    ptr++;  // Skip space

    // Field 5: Fullmove number
    field = 5;
    if (!*ptr || !isdigit(*ptr)) {
        fprintf(stderr, "FEN error: Expected digit for fullmove number\n");
        return false;
    }

    char fullmove_str[4] = {0};
    i = 0;
    while (*ptr && isdigit(*ptr) && i < 3) {
        fullmove_str[i++] = *ptr++;
    }
    board->fullmove_number = atoi(fullmove_str);

    // Check that all pieces are on the board
    bool has_white_king = false;
    bool has_black_king = false;

    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type == KING) {
            if (board->pieces[sq].color == WHITE)
                has_white_king = true;
            else
                has_black_king = true;
        }
    }

    if (!has_white_king || !has_black_king) {
        fprintf(stderr, "FEN error: Missing king(s)\n");
        return false;
    }

    return true;
}

// Convert board state to FEN string
char *board_to_fen(const Board *board) {
    static char fen[100];
    char *ptr = fen;

    // 1. Piece placement
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;

        for (int file = 0; file < 8; file++) {
            int square = SQUARE(file, rank);
            Piece piece = board->pieces[square];

            if (piece.type == EMPTY) {
                empty++;
            } else {
                if (empty > 0) {
                    *ptr++ = '0' + empty;
                    empty = 0;
                }

                char pc;
                switch (piece.type) {
                case PAWN:
                    pc = 'p';
                    break;
                case KNIGHT:
                    pc = 'n';
                    break;
                case BISHOP:
                    pc = 'b';
                    break;
                case ROOK:
                    pc = 'r';
                    break;
                case QUEEN:
                    pc = 'q';
                    break;
                case KING:
                    pc = 'k';
                    break;
                default:
                    pc = '?';
                    break;
                }

                if (piece.color == WHITE) pc = toupper(pc);
                *ptr++ = pc;
            }
        }

        if (empty > 0) *ptr++ = '0' + empty;

        if (rank > 0) *ptr++ = '/';
    }

    // 2. Side to move
    *ptr++ = ' ';
    *ptr++ = (board->side_to_move == WHITE) ? 'w' : 'b';

    // 3. Castling rights
    *ptr++ = ' ';
    bool has_castle = false;
    if (board->castle_rights & CASTLE_WHITE_KINGSIDE) {
        *ptr++ = 'K';
        has_castle = true;
    }
    if (board->castle_rights & CASTLE_WHITE_QUEENSIDE) {
        *ptr++ = 'Q';
        has_castle = true;
    }
    if (board->castle_rights & CASTLE_BLACK_KINGSIDE) {
        *ptr++ = 'k';
        has_castle = true;
    }
    if (board->castle_rights & CASTLE_BLACK_QUEENSIDE) {
        *ptr++ = 'q';
        has_castle = true;
    }
    if (!has_castle) *ptr++ = '-';

    // 4. En passant square
    *ptr++ = ' ';
    if (board->en_passant_square == -1) {
        *ptr++ = '-';
    } else {
        *ptr++ = FILE(board->en_passant_square) + 'a';
        *ptr++ = RANK(board->en_passant_square) + '1';
    }

    // 5. Halfmove clock
    sprintf(ptr, " %d %d", board->halfmove_clock, board->fullmove_number);

    return fen;
}

// Print the chess board
void print_board(const Board *board) {
    printf("  +----------------+\n");

    for (int rank = 7; rank >= 0; rank--) {
        printf("%d | ", rank + 1);

        for (int file = 0; file < 8; file++) {
            int square = SQUARE(file, rank);
            Piece piece = board->pieces[square];

            char pc;
            if (piece.type == EMPTY) {
                pc = '.';
            } else {
                switch (piece.type) {
                case PAWN:
                    pc = 'p';
                    break;
                case KNIGHT:
                    pc = 'n';
                    break;
                case BISHOP:
                    pc = 'b';
                    break;
                case ROOK:
                    pc = 'r';
                    break;
                case QUEEN:
                    pc = 'q';
                    break;
                case KING:
                    pc = 'k';
                    break;
                default:
                    pc = '?';
                    break;
                }

                if (piece.color == WHITE) pc = toupper(pc);
            }

            printf("%c ", pc);
        }

        printf("|\n");
    }

    printf("  +----------------+\n");
    printf("    a b c d e f g h\n");

    printf("Side to move: %s\n", board->side_to_move == WHITE ? "White" : "Black");

    printf("Castling: %s%s%s%s\n",
           (board->castle_rights & CASTLE_WHITE_KINGSIDE) ? "K" : "",
           (board->castle_rights & CASTLE_WHITE_QUEENSIDE) ? "Q" : "",
           (board->castle_rights & CASTLE_BLACK_KINGSIDE) ? "k" : "",
           (board->castle_rights & CASTLE_BLACK_QUEENSIDE) ? "q" : "");

    if (board->en_passant_square != -1) {
        char ep_file = FILE(board->en_passant_square) + 'a';
        char ep_rank = RANK(board->en_passant_square) + '1';
        printf("En passant: %c%c\n", ep_file, ep_rank);
    } else {
        printf("En passant: -\n");
    }

    printf("Halfmove clock: %d\n", board->halfmove_clock);
    printf("Fullmove number: %d\n", board->fullmove_number);
}

// Get the piece at a given square
Piece get_piece(const Board *board, int square) {
    return board->pieces[square];
}

// Convert a square index to a bitboard
Bitboard square_to_bitboard(int square) {
    return 1ULL << square;
}

// Find the square of a bitboard with a single bit set
int bitboard_to_square(Bitboard bb) {
    // De Bruijn sequence based bit scan
    static const int index64[64] = {
        0, 1, 2, 7, 3, 13, 8, 19,
        4, 25, 14, 28, 9, 34, 20, 40,
        5, 17, 26, 38, 15, 46, 29, 48,
        10, 31, 35, 54, 21, 50, 41, 57,
        63, 6, 12, 18, 24, 27, 33, 39,
        16, 37, 45, 47, 30, 53, 49, 56,
        62, 11, 23, 32, 36, 44, 52, 55,
        61, 22, 43, 51, 60, 42, 59, 58};

    const Bitboard debruijn64 = 0x03f79d71b4cb0a89ULL;
    return index64[((bb & -bb) * debruijn64) >> 58];
}

// Count the number of set bits in a bitboard
int count_bits(Bitboard bb) {
    int count = 0;
    while (bb) {
        count++;
        bb &= bb - 1;  // Clear the least significant bit
    }
    return count;
}

// Remove and return the index of the least significant bit
int pop_lsb(Bitboard *bb) {
    int square = bitboard_to_square(*bb);
    *bb &= *bb - 1;  // Clear the least significant bit
    return square;
}

// Determine if a square is attacked by a side
bool is_square_attacked(const Board *board, int square, Color by_side) {
    int from_file = FILE(square);
    int from_rank = RANK(square);

    // Pawn attacks
    if (by_side == WHITE) {
        // Check if white pawns attack this square
        if (from_rank > 0) {
            if (from_file > 0) {
                int pawn_sq = SQUARE(from_file - 1, from_rank - 1);
                if (board->pieces[pawn_sq].type == PAWN && board->pieces[pawn_sq].color == WHITE) {
                    return true;
                }
            }
            if (from_file < 7) {
                int pawn_sq = SQUARE(from_file + 1, from_rank - 1);
                if (board->pieces[pawn_sq].type == PAWN && board->pieces[pawn_sq].color == WHITE) {
                    return true;
                }
            }
        }
    } else {
        // Check if black pawns attack this square
        if (from_rank < 7) {
            if (from_file > 0) {
                int pawn_sq = SQUARE(from_file - 1, from_rank + 1);
                if (board->pieces[pawn_sq].type == PAWN && board->pieces[pawn_sq].color == BLACK) {
                    return true;
                }
            }
            if (from_file < 7) {
                int pawn_sq = SQUARE(from_file + 1, from_rank + 1);
                if (board->pieces[pawn_sq].type == PAWN && board->pieces[pawn_sq].color == BLACK) {
                    return true;
                }
            }
        }
    }

    // Knight attacks
    int knight_offsets[8][2] = {
        {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};

    for (int i = 0; i < 8; i++) {
        int to_file = from_file + knight_offsets[i][0];
        int to_rank = from_rank + knight_offsets[i][1];

        if (to_file >= 0 && to_file < 8 && to_rank >= 0 && to_rank < 8) {
            int to = SQUARE(to_file, to_rank);
            if (board->pieces[to].type == KNIGHT && board->pieces[to].color == by_side) {
                return true;
            }
        }
    }

    // King attacks
    int king_offsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    for (int i = 0; i < 8; i++) {
        int to_file = from_file + king_offsets[i][0];
        int to_rank = from_rank + king_offsets[i][1];

        if (to_file >= 0 && to_file < 8 && to_rank >= 0 && to_rank < 8) {
            int to = SQUARE(to_file, to_rank);
            if (board->pieces[to].type == KING && board->pieces[to].color == by_side) {
                return true;
            }
        }
    }

    // Bishop and Queen diagonal attacks
    int bishop_dirs[4][2] = {
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

    for (int i = 0; i < 4; i++) {
        int dir_file = bishop_dirs[i][0];
        int dir_rank = bishop_dirs[i][1];

        int to_file = from_file + dir_file;
        int to_rank = from_rank + dir_rank;

        while (to_file >= 0 && to_file < 8 && to_rank >= 0 && to_rank < 8) {
            int to = SQUARE(to_file, to_rank);
            Piece piece = board->pieces[to];

            if (piece.type != EMPTY) {
                if ((piece.type == BISHOP || piece.type == QUEEN) && piece.color == by_side) {
                    return true;
                }
                break;  // Blocked by a piece
            }

            to_file += dir_file;
            to_rank += dir_rank;
        }
    }

    // Rook and Queen orthogonal attacks
    int rook_dirs[4][2] = {
        {0, -1}, {-1, 0}, {1, 0}, {0, 1}};

    for (int i = 0; i < 4; i++) {
        int dir_file = rook_dirs[i][0];
        int dir_rank = rook_dirs[i][1];

        int to_file = from_file + dir_file;
        int to_rank = from_rank + dir_rank;

        while (to_file >= 0 && to_file < 8 && to_rank >= 0 && to_rank < 8) {
            int to = SQUARE(to_file, to_rank);
            Piece piece = board->pieces[to];

            if (piece.type != EMPTY) {
                if ((piece.type == ROOK || piece.type == QUEEN) && piece.color == by_side) {
                    return true;
                }
                break;  // Blocked by a piece
            }

            to_file += dir_file;
            to_rank += dir_rank;
        }
    }

    return false;
}

// Add a new function to validate board state consistency
void validate_board_state(Board *board) {
    // Clear all bitboards
    for (int color = 0; color < 2; color++) {
        board->occupied[color] = 0;
        for (int piece = 1; piece <= 6; piece++) {
            board->piece_bb[color][piece] = 0;
        }
    }

    // Rebuild bitboards from piece array
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board->pieces[sq];
        if (p.type != EMPTY) {
            Bitboard sq_bb = square_to_bitboard(sq);
            board->piece_bb[p.color][p.type] |= sq_bb;
            board->occupied[p.color] |= sq_bb;
        }
    }

    // Update all_pieces
    board->all_pieces = board->occupied[WHITE] | board->occupied[BLACK];
}
