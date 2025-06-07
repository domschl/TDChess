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
    board->king_pos[WHITE] = -1; // Initialize king positions to an invalid square
    board->king_pos[BLACK] = -1;
}

// Set up the standard chess starting position
void setup_default_position(Board *board) {
    // Start with an empty board
    init_board(board);

    // Standard chess starting position in FEN
    const char *start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // Parse the FEN string to set up the board
    if (!parse_fen(board, start_fen)) {
        fprintf(stderr, "Error: Failed to set up default position from FEN. Falling back.\n");

        // Fall back to minimal position with just kings
        init_board(board); // Re-init to clear any partial FEN parse

        // Place kings in their starting positions
        int white_king_sq = SQUARE(SQUARE_FILE_E, SQUARE_RANK_1);
        int black_king_sq = SQUARE(SQUARE_FILE_E, SQUARE_RANK_8);

        board->pieces[white_king_sq].type = KING;
        board->pieces[white_king_sq].color = WHITE;
        board->king_pos[WHITE] = white_king_sq;

        board->pieces[black_king_sq].type = KING;
        board->pieces[black_king_sq].color = BLACK;
        board->king_pos[BLACK] = black_king_sq;


        // Update bitboards
        board->piece_bb[WHITE][KING] = square_to_bitboard(white_king_sq);
        board->piece_bb[BLACK][KING] = square_to_bitboard(black_king_sq);
        board->occupied[WHITE] = board->piece_bb[WHITE][KING];
        board->occupied[BLACK] = board->piece_bb[BLACK][KING];
        board->all_pieces = board->occupied[WHITE] | board->occupied[BLACK];
    }

    // Double check that bitboards are consistent with piece positions
    // and that king_pos is correctly set.
    validate_board_state(board); // This should now also verify king_pos if you extend it.
                                 // For now, we ensure parse_fen sets it.
    if (board->king_pos[WHITE] == -1 || board->king_pos[BLACK] == -1) {
         fprintf(stderr, "Error: King positions not set after setup_default_position.\n");
         // Attempt to find kings if not set by FEN (e.g. if FEN was invalid but fallback didn't run)
         for(int i=0; i<64; ++i) {
            if(board->pieces[i].type == KING) {
                board->king_pos[board->pieces[i].color] = i;
            }
         }
         if (board->king_pos[WHITE] == -1 || board->king_pos[BLACK] == -1) {
            fprintf(stderr, "Critical Error: Could not find kings on board after setup.\n");
            // Handle this critical error, perhaps by exiting or using a known safe state.
         }
    }
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
                // Store king position
                board->king_pos[piece.color] = SQUARE(file, rank);
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
    if (board->king_pos[WHITE] == -1 || board->king_pos[BLACK] == -1) {
        fprintf(stderr, "FEN error: Missing king(s) or king_pos not set.\n");
        return false;
    }

    return true;
}

// Update the implementation
bool board_to_fen(const Board *board, char *buffer, size_t buffer_size) {
    if (!buffer || buffer_size < 90) {  // FEN strings can be quite long
        return false;
    }

    // Implementation of FEN generation
    int offset = 0;

    // 1. Piece placement
    for (int rank = 7; rank >= 0; rank--) {
        int empty_count = 0;

        for (int file = 0; file < 8; file++) {
            int square = rank * 8 + file;
            Piece piece = board->pieces[square];

            if (piece.type == EMPTY) {
                empty_count++;
            } else {
                // If there were empty squares before this piece, add the count
                if (empty_count > 0) {
                    offset += snprintf(buffer + offset, buffer_size - offset, "%d", empty_count);
                    empty_count = 0;
                }

                // Add the piece character
                char piece_char = get_piece_char(piece);
                offset += snprintf(buffer + offset, buffer_size - offset, "%c", piece_char);
            }
        }

        // If there are empty squares at the end of the rank, add the count
        if (empty_count > 0) {
            offset += snprintf(buffer + offset, buffer_size - offset, "%d", empty_count);
        }

        // Add rank separator (except for the last rank)
        if (rank > 0) {
            offset += snprintf(buffer + offset, buffer_size - offset, "/");
        }
    }

    // 2. Active color
    offset += snprintf(buffer + offset, buffer_size - offset, " %c",
                       board->side_to_move == WHITE ? 'w' : 'b');

    // 3. Castling availability
    offset += snprintf(buffer + offset, buffer_size - offset, " ");
    bool has_castling = false;

    if (board->castle_rights & CASTLE_WHITE_KINGSIDE) {
        offset += snprintf(buffer + offset, buffer_size - offset, "K");
        has_castling = true;
    }
    if (board->castle_rights & CASTLE_WHITE_QUEENSIDE) {
        offset += snprintf(buffer + offset, buffer_size - offset, "Q");
        has_castling = true;
    }
    if (board->castle_rights & CASTLE_BLACK_KINGSIDE) {
        offset += snprintf(buffer + offset, buffer_size - offset, "k");
        has_castling = true;
    }
    if (board->castle_rights & CASTLE_BLACK_QUEENSIDE) {
        offset += snprintf(buffer + offset, buffer_size - offset, "q");
        has_castling = true;
    }

    if (!has_castling) {
        offset += snprintf(buffer + offset, buffer_size - offset, "-");
    }

    // 4. En passant target square
    offset += snprintf(buffer + offset, buffer_size - offset, " ");
    if (board->en_passant_square == -1) {
        offset += snprintf(buffer + offset, buffer_size - offset, "-");
    } else {
        int ep_file = board->en_passant_square % 8;
        int ep_rank = board->en_passant_square / 8;
        offset += snprintf(buffer + offset, buffer_size - offset, "%c%d",
                           'a' + ep_file, ep_rank + 1);
    }

    // 5. Halfmove clock
    offset += snprintf(buffer + offset, buffer_size - offset, " %d", board->halfmove_clock);

    // 6. Fullmove number
    offset += snprintf(buffer + offset, buffer_size - offset, " %d", board->fullmove_number);

    return true;
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
    int from_file = SQUARE_FILE(square);
    int from_rank = SQUARE_RANK(square);

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

// Convert a piece to its character representation
char get_piece_char(Piece piece) {
    if (piece.type == EMPTY) {
        return '.';
    }

    char piece_chars[] = "PNBRQK";
    char c = piece_chars[piece.type - 1];

    return (piece.color == WHITE) ? c : (char)(c + ('a' - 'A'));
}
