#include "movegen.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

// Initialize a move list
static void init_move_list(MoveList *list) {
    list->count = 0;
}

// Add a move to the move list
void add_move(MoveList *list, int from, int to, PieceType promotion, bool capture, bool castling, bool en_passant, int captured_square) {
    if (list->count < MAX_MOVES) {
        Move *move = &list->moves[list->count++];
        move->from = from;
        move->to = to;
        move->promotion = promotion;
        move->capture = capture;
        move->castling = castling;
        move->en_passant = en_passant;
        move->captured_piece_square = captured_square;
    }
}

// Generate all pseudo-legal moves for a position
void generate_moves(const Board *board, MoveList *list) {
    init_move_list(list);

    generate_pawn_moves(board, list);
    generate_knight_moves(board, list);
    generate_bishop_moves(board, list);
    generate_rook_moves(board, list);
    generate_queen_moves(board, list);
    generate_king_moves(board, list);
}

// Generate all legally valid moves
void generate_legal_moves(const Board *board, MoveList *list) {
    MoveList pseudo_legal;
    generate_moves(board, &pseudo_legal);

    init_move_list(list);

    for (int i = 0; i < pseudo_legal.count; i++) {
        if (is_move_legal(board, pseudo_legal.moves[i])) {
            list->moves[list->count++] = pseudo_legal.moves[i];
        }
    }
}

// Check if a move is legal (does not leave the king in check)
bool is_move_legal(const Board *board, Move move) {
    // Make a copy of the board
    Board temp_board = *board;

    // Make the move on the copy
    make_move(&temp_board, move);

    // Find the king of the side that just moved
    Color side = board->side_to_move;
    int king_square = -1;

    // Search for the king
    for (int sq = 0; sq < 64; sq++) {
        if (temp_board.pieces[sq].type == KING && temp_board.pieces[sq].color == side) {
            king_square = sq;
            break;
        }
    }

    // If king not found (shouldn't happen), consider the move illegal
    if (king_square == -1) {
        return false;
    }

    // Check if the king is in check after the move
    if (is_square_attacked(&temp_board, king_square, !side)) {
        return false;  // King would be in check, illegal move
    }

    return true;  // Move is legal
}

// Generate pawn moves
void generate_pawn_moves(const Board *board, MoveList *list) {
    Color side = board->side_to_move;
    Color opponent = !side;

    // Direction pawns move (up for white, down for black)
    int pawn_push = (side == WHITE) ? 8 : -8;
    int start_rank = (side == WHITE) ? RANK_2 : RANK_7;
    int promotion_rank = (side == WHITE) ? RANK_7 : RANK_2;

    // Get all pawns of the current side
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type != PAWN || board->pieces[sq].color != side) {
            continue;
        }

        int file = FILE(sq);
        int rank = RANK(sq);

        // Single push
        int to_sq = sq + pawn_push;
        if (to_sq >= 0 && to_sq < 64 && board->pieces[to_sq].type == EMPTY) {
            // Check for promotion
            if (rank == promotion_rank) {
                add_move(list, sq, to_sq, QUEEN, false, false, false, -1);
                add_move(list, sq, to_sq, ROOK, false, false, false, -1);
                add_move(list, sq, to_sq, BISHOP, false, false, false, -1);
                add_move(list, sq, to_sq, KNIGHT, false, false, false, -1);
            } else {
                add_move(list, sq, to_sq, EMPTY, false, false, false, -1);

                // Double push from starting rank
                if (rank == start_rank) {
                    int double_to = sq + 2 * pawn_push;
                    if (double_to >= 0 && double_to < 64 && board->pieces[double_to].type == EMPTY) {
                        add_move(list, sq, double_to, EMPTY, false, false, false, -1);
                    }
                }
            }
        }

        // Captures
        for (int dir = -1; dir <= 1; dir += 2) {
            // Skip if at the edge of the board
            if ((dir == -1 && file == 0) || (dir == 1 && file == 7)) {
                continue;
            }

            int capture_sq = sq + pawn_push + dir;

            // Ensure the square is on the board
            if (capture_sq < 0 || capture_sq >= 64) {
                continue;
            }

            // Regular capture
            if (board->pieces[capture_sq].type != EMPTY &&
                board->pieces[capture_sq].color == opponent) {

                // Check for promotion
                if (rank == promotion_rank) {
                    add_move(list, sq, capture_sq, QUEEN, true, false, false, -1);
                    add_move(list, sq, capture_sq, ROOK, true, false, false, -1);
                    add_move(list, sq, capture_sq, BISHOP, true, false, false, -1);
                    add_move(list, sq, capture_sq, KNIGHT, true, false, false, -1);
                } else {
                    add_move(list, sq, capture_sq, EMPTY, true, false, false, -1);
                }
            }

            // En passant capture
            if (capture_sq == board->en_passant_square && board->en_passant_square != -1) {
                int captured_sq = capture_sq - pawn_push;  // Pawn to be captured
                add_move(list, sq, capture_sq, EMPTY, true, false, true, captured_sq);
            }
        }
    }
}

// Generate knight moves
void generate_knight_moves(const Board *board, MoveList *list) {
    Color side = board->side_to_move;

    // Knight movement offsets
    int knight_dirs[8] = {-17, -15, -10, -6, 6, 10, 15, 17};

    for (int square = 0; square < 64; square++) {
        if (board->pieces[square].type == KNIGHT && board->pieces[square].color == side) {
            for (int i = 0; i < 8; i++) {
                int to = square + knight_dirs[i];

                // Check if the target square is on the board
                if (to < 0 || to >= 64) continue;

                // Check if the knight's move is valid (not wrapping around the board)
                int from_file = FILE(square);
                int from_rank = RANK(square);
                int to_file = FILE(to);
                int to_rank = RANK(to);

                int file_diff = abs(from_file - to_file);
                int rank_diff = abs(from_rank - to_rank);

                if (!((file_diff == 1 && rank_diff == 2) || (file_diff == 2 && rank_diff == 1))) {
                    continue;
                }

                // Empty square or capture
                if (board->pieces[to].type == EMPTY) {
                    add_move(list, square, to, EMPTY, false, false, false, -1);
                } else if (board->pieces[to].color != side) {
                    add_move(list, square, to, EMPTY, true, false, false, -1);
                }
            }
        }
    }
}

// Generate bishop moves
void generate_bishop_moves(const Board *board, MoveList *list) {
    Color side = board->side_to_move;

    // Bishop movement directions (diagonal)
    int bishop_dirs[4] = {-9, -7, 7, 9};

    for (int square = 0; square < 64; square++) {
        if (board->pieces[square].type == BISHOP && board->pieces[square].color == side) {
            int from_file = FILE(square);
            int from_rank = RANK(square);

            for (int i = 0; i < 4; i++) {
                int dir = bishop_dirs[i];
                int to = square;
                int current_file = from_file;
                int current_rank = from_rank;

                while (true) {
                    to += dir;

                    // Check if the target square is on the board
                    if (to < 0 || to >= 64) break;

                    int to_file = FILE(to);
                    int to_rank = RANK(to);

                    // Check if we've moved diagonally (file and rank should change by the same amount)
                    int file_diff = abs(to_file - current_file);
                    int rank_diff = abs(to_rank - current_rank);

                    if (file_diff != 1 || rank_diff != 1) break;

                    current_file = to_file;
                    current_rank = to_rank;

                    // Empty square
                    if (board->pieces[to].type == EMPTY) {
                        add_move(list, square, to, EMPTY, false, false, false, -1);
                    }
                    // Capture
                    else if (board->pieces[to].color != side) {
                        add_move(list, square, to, EMPTY, true, false, false, -1);
                        break;  // Stop after capturing
                    }
                    // Own piece
                    else {
                        break;  // Blocked by own piece
                    }
                }
            }
        }
    }
}

// Generate rook moves
void generate_rook_moves(const Board *board, MoveList *list) {
    Color side = board->side_to_move;

    // Rook movement directions (orthogonal)
    int rook_dirs[4] = {-8, -1, 1, 8};

    for (int square = 0; square < 64; square++) {
        if (board->pieces[square].type == ROOK && board->pieces[square].color == side) {
            int from_file = FILE(square);
            int from_rank = RANK(square);

            for (int i = 0; i < 4; i++) {
                int dir = rook_dirs[i];
                int to = square;
                int current_file = from_file;
                int current_rank = from_rank;

                while (true) {
                    to += dir;

                    // Check if the target square is on the board
                    if (to < 0 || to >= 64) break;

                    int to_file = FILE(to);
                    int to_rank = RANK(to);

                    // Check if we're moving horizontally or vertically (not wrapping around the board)
                    if (dir == 1 || dir == -1) {  // Horizontal movement
                        if (to_rank != current_rank) break;
                    } else {  // Vertical movement
                        if (to_file != current_file) break;
                    }

                    current_file = to_file;
                    current_rank = to_rank;

                    // Empty square
                    if (board->pieces[to].type == EMPTY) {
                        add_move(list, square, to, EMPTY, false, false, false, -1);
                    }
                    // Capture
                    else if (board->pieces[to].color != side) {
                        add_move(list, square, to, EMPTY, true, false, false, -1);
                        break;  // Stop after capturing
                    }
                    // Own piece
                    else {
                        break;  // Blocked by own piece
                    }
                }
            }
        }
    }
}

// Generate queen moves (combination of bishop and rook moves)
void generate_queen_moves(const Board *board, MoveList *list) {
    Color side = board->side_to_move;

    // Queen movement directions (both diagonal and orthogonal)
    int queen_dirs[8] = {-9, -8, -7, -1, 1, 7, 8, 9};

    for (int square = 0; square < 64; square++) {
        if (board->pieces[square].type == QUEEN && board->pieces[square].color == side) {
            int from_file = FILE(square);
            int from_rank = RANK(square);

            for (int i = 0; i < 8; i++) {
                int dir = queen_dirs[i];
                int to = square;
                int current_file = from_file;
                int current_rank = from_rank;

                while (true) {
                    to += dir;

                    // Check if the target square is on the board
                    if (to < 0 || to >= 64) break;

                    int to_file = FILE(to);
                    int to_rank = RANK(to);

                    // Check move validity based on direction
                    if (dir == -9 || dir == 7 || dir == -7 || dir == 9) {  // Diagonal moves
                        int file_diff = abs(to_file - current_file);
                        int rank_diff = abs(to_rank - current_rank);
                        if (file_diff != 1 || rank_diff != 1) break;
                    } else if (dir == -1 || dir == 1) {  // Horizontal moves
                        if (to_rank != current_rank) break;
                    } else {  // Vertical moves
                        if (to_file != current_file) break;
                    }

                    current_file = to_file;
                    current_rank = to_rank;

                    // Empty square
                    if (board->pieces[to].type == EMPTY) {
                        add_move(list, square, to, EMPTY, false, false, false, -1);
                    }
                    // Capture
                    else if (board->pieces[to].color != side) {
                        add_move(list, square, to, EMPTY, true, false, false, -1);
                        break;  // Stop after capturing
                    }
                    // Own piece
                    else {
                        break;  // Blocked by own piece
                    }
                }
            }
        }
    }
}

// Generate king moves
void generate_king_moves(const Board *board, MoveList *list) {
    Color side = board->side_to_move;

    for (int square = 0; square < 64; square++) {
        if (board->pieces[square].type == KING && board->pieces[square].color == side) {
            // King movement directions (all 8 surrounding squares)
            int king_dirs[8] = {-9, -8, -7, -1, 1, 7, 8, 9};

            for (int i = 0; i < 8; i++) {
                int to = square + king_dirs[i];

                // Check if the target square is on the board
                if (to < 0 || to >= 64) continue;

                // Check if the move doesn't wrap around the board
                int from_file = FILE(square);
                int from_rank = RANK(square);
                int to_file = FILE(to);
                int to_rank = RANK(to);

                int file_diff = abs(from_file - to_file);
                int rank_diff = abs(from_rank - to_rank);

                if (file_diff > 1 || rank_diff > 1) continue;

                // Empty square or capture
                if (board->pieces[to].type == EMPTY) {
                    add_move(list, square, to, EMPTY, false, false, false, -1);
                } else if (board->pieces[to].color != side) {
                    add_move(list, square, to, EMPTY, true, false, false, -1);
                }
            }

            // Castling
            if (side == WHITE) {
                // White kingside castling
                if ((board->castle_rights & CASTLE_WHITE_KINGSIDE) &&
                    board->pieces[SQUARE(FILE_F, RANK_1)].type == EMPTY &&
                    board->pieces[SQUARE(FILE_G, RANK_1)].type == EMPTY) {

                    // Check if king and path to castling are not in check
                    if (!is_square_attacked(board, square, BLACK) &&
                        !is_square_attacked(board, SQUARE(FILE_F, RANK_1), BLACK)) {
                        add_move(list, square, SQUARE(FILE_G, RANK_1), EMPTY, false, true, false, -1);
                    }
                }

                // White queenside castling
                if ((board->castle_rights & CASTLE_WHITE_QUEENSIDE) &&
                    board->pieces[SQUARE(FILE_D, RANK_1)].type == EMPTY &&
                    board->pieces[SQUARE(FILE_C, RANK_1)].type == EMPTY &&
                    board->pieces[SQUARE(FILE_B, RANK_1)].type == EMPTY) {

                    // Check if king and path to castling are not in check
                    if (!is_square_attacked(board, square, BLACK) &&
                        !is_square_attacked(board, SQUARE(FILE_D, RANK_1), BLACK)) {
                        add_move(list, square, SQUARE(FILE_C, RANK_1), EMPTY, false, true, false, -1);
                    }
                }
            } else {
                // Black kingside castling
                if ((board->castle_rights & CASTLE_BLACK_KINGSIDE) &&
                    board->pieces[SQUARE(FILE_F, RANK_8)].type == EMPTY &&
                    board->pieces[SQUARE(FILE_G, RANK_8)].type == EMPTY) {

                    // Check if king and path to castling are not in check
                    if (!is_square_attacked(board, square, WHITE) &&
                        !is_square_attacked(board, SQUARE(FILE_F, RANK_8), WHITE)) {
                        add_move(list, square, SQUARE(FILE_G, RANK_8), EMPTY, false, true, false, -1);
                    }
                }

                // Black queenside castling
                if ((board->castle_rights & CASTLE_BLACK_QUEENSIDE) &&
                    board->pieces[SQUARE(FILE_D, RANK_8)].type == EMPTY &&
                    board->pieces[SQUARE(FILE_C, RANK_8)].type == EMPTY &&
                    board->pieces[SQUARE(FILE_B, RANK_8)].type == EMPTY) {

                    // Check if king and path to castling are not in check
                    if (!is_square_attacked(board, square, WHITE) &&
                        !is_square_attacked(board, SQUARE(FILE_D, RANK_8), WHITE)) {
                        add_move(list, square, SQUARE(FILE_C, RANK_8), EMPTY, false, true, false, -1);
                    }
                }
            }

            break;  // We found the king, no need to continue
        }
    }
}

// Make a move on the board
void make_move(Board *board, Move move) {
    int from = move.from;
    int to = move.to;

    Piece piece = board->pieces[from];
    Piece captured_piece = {EMPTY, 0};

    // Handle castling
    if (move.castling) {
        // Move the king
        board->pieces[to] = board->pieces[from];
        board->pieces[from].type = EMPTY;

        // Update bitboards for the king
        Bitboard from_bb = square_to_bitboard(from);
        Bitboard to_bb = square_to_bitboard(to);
        board->piece_bb[piece.color][KING] &= ~from_bb;
        board->piece_bb[piece.color][KING] |= to_bb;
        board->occupied[piece.color] &= ~from_bb;
        board->occupied[piece.color] |= to_bb;

        // Move the rook
        int rook_from, rook_to;
        if (to > from) {
            // Kingside castling
            rook_from = (piece.color == WHITE) ? SQUARE(FILE_H, RANK_1) : SQUARE(FILE_H, RANK_8);
            rook_to = (piece.color == WHITE) ? SQUARE(FILE_F, RANK_1) : SQUARE(FILE_F, RANK_8);
        } else {
            // Queenside castling
            rook_from = (piece.color == WHITE) ? SQUARE(FILE_A, RANK_1) : SQUARE(FILE_A, RANK_8);
            rook_to = (piece.color == WHITE) ? SQUARE(FILE_D, RANK_1) : SQUARE(FILE_D, RANK_8);
        }

        board->pieces[rook_to] = board->pieces[rook_from];
        board->pieces[rook_from].type = EMPTY;

        // Update bitboards for the rook
        Bitboard rook_from_bb = square_to_bitboard(rook_from);
        Bitboard rook_to_bb = square_to_bitboard(rook_to);
        board->piece_bb[piece.color][ROOK] &= ~rook_from_bb;
        board->piece_bb[piece.color][ROOK] |= rook_to_bb;
        board->occupied[piece.color] &= ~rook_from_bb;
        board->occupied[piece.color] |= rook_to_bb;
    }
    // Handle en passant capture
    else if (move.en_passant) {
        // Move the pawn
        board->pieces[to] = board->pieces[from];
        board->pieces[from].type = EMPTY;

        // Update bitboards for the pawn
        Bitboard from_bb = square_to_bitboard(from);
        Bitboard to_bb = square_to_bitboard(to);
        board->piece_bb[piece.color][PAWN] &= ~from_bb;
        board->piece_bb[piece.color][PAWN] |= to_bb;
        board->occupied[piece.color] &= ~from_bb;
        board->occupied[piece.color] |= to_bb;

        // Capture the opponent's pawn
        int captured = move.captured_piece_square;
        captured_piece = board->pieces[captured];
        board->pieces[captured].type = EMPTY;

        // Update bitboards for the captured pawn
        Bitboard captured_bb = square_to_bitboard(captured);
        board->piece_bb[captured_piece.color][PAWN] &= ~captured_bb;
        board->occupied[captured_piece.color] &= ~captured_bb;
    }
    // Handle regular move or capture
    else {
        // Check for capture
        if (move.capture) {
            captured_piece = board->pieces[to];

            // Update bitboards for the captured piece
            Bitboard to_bb = square_to_bitboard(to);
            board->piece_bb[captured_piece.color][captured_piece.type] &= ~to_bb;
            board->occupied[captured_piece.color] &= ~to_bb;
        }

        // Move the piece
        if (move.promotion != EMPTY) {
            board->pieces[to].type = move.promotion;
            board->pieces[to].color = piece.color;
        } else {
            board->pieces[to] = piece;
        }
        board->pieces[from].type = EMPTY;

        // Update bitboards
        Bitboard from_bb = square_to_bitboard(from);
        Bitboard to_bb = square_to_bitboard(to);

        // Clear the piece from the 'from' square
        board->piece_bb[piece.color][piece.type] &= ~from_bb;
        board->occupied[piece.color] &= ~from_bb;

        // Set the piece on the 'to' square
        PieceType final_type = move.promotion != EMPTY ? move.promotion : piece.type;
        board->piece_bb[piece.color][final_type] |= to_bb;
        board->occupied[piece.color] |= to_bb;
    }

    // Update all pieces bitboard
    board->all_pieces = board->occupied[WHITE] | board->occupied[BLACK];

    // Update en passant square
    if (piece.type == PAWN && abs(to - from) == 16) {
        // Double pawn push
        board->en_passant_square = (from + to) / 2;  // The square in between
    } else {
        board->en_passant_square = -1;
    }

    // Update castling rights
    if (piece.type == KING) {
        if (piece.color == WHITE) {
            board->castle_rights &= ~(CASTLE_WHITE_KINGSIDE | CASTLE_WHITE_QUEENSIDE);
        } else {
            board->castle_rights &= ~(CASTLE_BLACK_KINGSIDE | CASTLE_BLACK_QUEENSIDE);
        }
    } else if (piece.type == ROOK) {
        if (from == SQUARE(FILE_A, RANK_1)) {
            board->castle_rights &= ~CASTLE_WHITE_QUEENSIDE;
        } else if (from == SQUARE(FILE_H, RANK_1)) {
            board->castle_rights &= ~CASTLE_WHITE_KINGSIDE;
        } else if (from == SQUARE(FILE_A, RANK_8)) {
            board->castle_rights &= ~CASTLE_BLACK_QUEENSIDE;
        } else if (from == SQUARE(FILE_H, RANK_8)) {
            board->castle_rights &= ~CASTLE_BLACK_KINGSIDE;
        }
    }

    // If a rook is captured, update castling rights
    if (captured_piece.type == ROOK) {
        if (to == SQUARE(FILE_A, RANK_1)) {
            board->castle_rights &= ~CASTLE_WHITE_QUEENSIDE;
        } else if (to == SQUARE(FILE_H, RANK_1)) {
            board->castle_rights &= ~CASTLE_WHITE_KINGSIDE;
        } else if (to == SQUARE(FILE_A, RANK_8)) {
            board->castle_rights &= ~CASTLE_BLACK_QUEENSIDE;
        } else if (to == SQUARE(FILE_H, RANK_8)) {
            board->castle_rights &= ~CASTLE_BLACK_KINGSIDE;
        }
    }

    // Update move counters
    if (piece.type == PAWN || move.capture) {
        board->halfmove_clock = 0;
    } else {
        board->halfmove_clock++;
    }

    if (piece.color == BLACK) {
        board->fullmove_number++;
    }

    // Switch sides
    board->side_to_move = !board->side_to_move;
}

// Convert move to algebraic notation
char *move_to_string(Move move) {
    static char str[6];

    char files[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
    char ranks[8] = {'1', '2', '3', '4', '5', '6', '7', '8'};

    int from_file = FILE(move.from);
    int from_rank = RANK(move.from);
    int to_file = FILE(move.to);
    int to_rank = RANK(move.to);

    str[0] = files[from_file];
    str[1] = ranks[from_rank];
    str[2] = files[to_file];
    str[3] = ranks[to_rank];

    if (move.promotion != EMPTY) {
        char promotion = '?';
        switch (move.promotion) {
        case QUEEN:
            promotion = 'q';
            break;
        case ROOK:
            promotion = 'r';
            break;
        case BISHOP:
            promotion = 'b';
            break;
        case KNIGHT:
            promotion = 'n';
            break;
        default:
            break;
        }
        str[4] = promotion;
        str[5] = '\0';
    } else {
        str[4] = '\0';
    }

    return str;
}

// Convert algebraic notation to move
Move string_to_move(const Board *board, const char *str) {
    Move move = {0};

    // Parse the move string
    int from_file = str[0] - 'a';
    int from_rank = str[1] - '1';
    int to_file = str[2] - 'a';
    int to_rank = str[3] - '1';

    move.from = SQUARE(from_file, from_rank);
    move.to = SQUARE(to_file, to_rank);

    // Check for promotion
    if (str[4] != '\0') {
        switch (str[4]) {
        case 'q':
            move.promotion = QUEEN;
            break;
        case 'r':
            move.promotion = ROOK;
            break;
        case 'b':
            move.promotion = BISHOP;
            break;
        case 'n':
            move.promotion = KNIGHT;
            break;
        default:
            break;
        }
    }

    // Determine if this is a capture, en passant, or castling
    Piece piece = board->pieces[move.from];

    // Check for capture
    if (board->pieces[move.to].type != EMPTY) {
        move.capture = true;
    }

    // Check for en passant
    if (piece.type == PAWN && move.to == board->en_passant_square) {
        move.en_passant = true;
        move.captured_piece_square = move.to + (piece.color == WHITE ? -8 : 8);
        move.capture = true;
    }

    // Check for castling
    if (piece.type == KING && abs(move.to - move.from) == 2) {
        move.castling = true;
    }

    return move;
}

// Generate all captures for the current side to move
void generate_captures(const Board *board, MoveList *list) {
    // Clear the move list
    list->count = 0;

    // Generate captures for each piece type
    generate_pawn_captures(board, list);
    generate_knight_captures(board, list);
    generate_bishop_captures(board, list);
    generate_rook_captures(board, list);
    generate_queen_captures(board, list);
    generate_king_captures(board, list);
}

// Generate pawn captures
void generate_pawn_captures(const Board *board, MoveList *list) {
    Color side = board->side_to_move;
    Color opponent = !side;

    // Direction pawns move (up for white, down for black)
    int pawn_push = (side == WHITE) ? 8 : -8;
    int promotion_rank = (side == WHITE) ? RANK_7 : RANK_2;

    // Get all pawns of the current side
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type != PAWN || board->pieces[sq].color != side) {
            continue;
        }

        int file = FILE(sq);
        int rank = RANK(sq);

        // Promotion is considered a "capture" for quiescence search
        if (rank == promotion_rank) {
            int to_sq = sq + pawn_push;
            if (to_sq >= 0 && to_sq < 64 && board->pieces[to_sq].type == EMPTY) {
                add_move(list, sq, to_sq, QUEEN, false, false, false, -1);
                // Only queen promotions for quiescence (simplification)
            }
        }

        // Captures
        for (int dir = -1; dir <= 1; dir += 2) {
            // Skip if at the edge of the board
            if ((dir == -1 && file == 0) || (dir == 1 && file == 7)) {
                continue;
            }

            int capture_sq = sq + pawn_push + dir;

            // Ensure the square is on the board
            if (capture_sq < 0 || capture_sq >= 64) {
                continue;
            }

            // Regular capture
            if (board->pieces[capture_sq].type != EMPTY &&
                board->pieces[capture_sq].color == opponent) {

                // Check for promotion
                if (rank == promotion_rank) {
                    add_move(list, sq, capture_sq, QUEEN, true, false, false, -1);
                    // Only queen promotions for quiescence (simplification)
                } else {
                    add_move(list, sq, capture_sq, EMPTY, true, false, false, -1);
                }
            }

            // En passant capture
            if (capture_sq == board->en_passant_square && board->en_passant_square != -1) {
                int captured_sq = capture_sq - pawn_push;  // Pawn to be captured
                add_move(list, sq, capture_sq, EMPTY, true, false, true, captured_sq);
            }
        }
    }
}

// Generate knight captures
void generate_knight_captures(const Board *board, MoveList *list) {
    Color side = board->side_to_move;
    Color opponent = !side;

    // Knight movement offsets
    int knight_offsets[8][2] = {
        {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};

    // Get all knights of the current side
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type != KNIGHT || board->pieces[sq].color != side) {
            continue;
        }

        int from_file = FILE(sq);
        int from_rank = RANK(sq);

        // Try each knight move
        for (int i = 0; i < 8; i++) {
            int to_file = from_file + knight_offsets[i][0];
            int to_rank = from_rank + knight_offsets[i][1];

            // Check if the target square is on the board
            if (to_file < 0 || to_file > 7 || to_rank < 0 || to_rank > 7) {
                continue;
            }

            int to_sq = SQUARE(to_file, to_rank);

            // Only consider captures
            if (board->pieces[to_sq].type != EMPTY && board->pieces[to_sq].color == opponent) {
                add_move(list, sq, to_sq, EMPTY, true, false, false, -1);
            }
        }
    }
}

// Generate bishop captures
void generate_bishop_captures(const Board *board, MoveList *list) {
    Color side = board->side_to_move;
    Color opponent = !side;

    // Bishop movement directions (diagonal)
    int bishop_dirs[4][2] = {
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
    };

    // Get all bishops of the current side
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type != BISHOP || board->pieces[sq].color != side) {
            continue;
        }

        int from_file = FILE(sq);
        int from_rank = RANK(sq);

        // Try each direction
        for (int i = 0; i < 4; i++) {
            int dir_file = bishop_dirs[i][0];
            int dir_rank = bishop_dirs[i][1];

            int to_file = from_file + dir_file;
            int to_rank = from_rank + dir_rank;

            // Move in this direction until we hit a piece or the edge of the board
            while (to_file >= 0 && to_file <= 7 && to_rank >= 0 && to_rank <= 7) {
                int to_sq = SQUARE(to_file, to_rank);

                // Empty square - continue sliding
                if (board->pieces[to_sq].type == EMPTY) {
                    to_file += dir_file;
                    to_rank += dir_rank;
                    continue;
                }

                // Capture opponent's piece
                if (board->pieces[to_sq].color == opponent) {
                    add_move(list, sq, to_sq, EMPTY, true, false, false, -1);
                }

                // Stop sliding after hitting any piece
                break;
            }
        }
    }
}

// Generate rook captures
void generate_rook_captures(const Board *board, MoveList *list) {
    Color side = board->side_to_move;
    Color opponent = !side;

    // Rook movement directions (orthogonal)
    int rook_dirs[4][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    // Get all rooks of the current side
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type != ROOK || board->pieces[sq].color != side) {
            continue;
        }

        int from_file = FILE(sq);
        int from_rank = RANK(sq);

        // Try each direction
        for (int i = 0; i < 4; i++) {
            int dir_file = rook_dirs[i][0];
            int dir_rank = rook_dirs[i][1];

            int to_file = from_file + dir_file;
            int to_rank = from_rank + dir_rank;

            // Move in this direction until we hit a piece or the edge of the board
            while (to_file >= 0 && to_file <= 7 && to_rank >= 0 && to_rank <= 7) {
                int to_sq = SQUARE(to_file, to_rank);

                // Empty square - continue sliding
                if (board->pieces[to_sq].type == EMPTY) {
                    to_file += dir_file;
                    to_rank += dir_rank;
                    continue;
                }

                // Capture opponent's piece
                if (board->pieces[to_sq].color == opponent) {
                    add_move(list, sq, to_sq, EMPTY, true, false, false, -1);
                }

                // Stop sliding after hitting any piece
                break;
            }
        }
    }
}

// Generate queen captures
void generate_queen_captures(const Board *board, MoveList *list) {
    Color side = board->side_to_move;
    Color opponent = !side;

    // Queen movement directions (both diagonal and orthogonal)
    int queen_dirs[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    // Get all queens of the current side
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type != QUEEN || board->pieces[sq].color != side) {
            continue;
        }

        int from_file = FILE(sq);
        int from_rank = RANK(sq);

        // Try each direction
        for (int i = 0; i < 8; i++) {
            int dir_file = queen_dirs[i][0];
            int dir_rank = queen_dirs[i][1];

            int to_file = from_file + dir_file;
            int to_rank = from_rank + dir_rank;

            // Move in this direction until we hit a piece or the edge of the board
            while (to_file >= 0 && to_file <= 7 && to_rank >= 0 && to_rank <= 7) {
                int to_sq = SQUARE(to_file, to_rank);

                // Empty square - continue sliding
                if (board->pieces[to_sq].type == EMPTY) {
                    to_file += dir_file;
                    to_rank += dir_rank;
                    continue;
                }

                // Capture opponent's piece
                if (board->pieces[to_sq].color == opponent) {
                    add_move(list, sq, to_sq, EMPTY, true, false, false, -1);
                }

                // Stop sliding after hitting any piece
                break;
            }
        }
    }
}

// Generate king captures
void generate_king_captures(const Board *board, MoveList *list) {
    Color side = board->side_to_move;
    Color opponent = !side;

    // King movement directions
    int king_dirs[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    // Find the king
    int king_sq = -1;
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type == KING && board->pieces[sq].color == side) {
            king_sq = sq;
            break;
        }
    }

    if (king_sq == -1) return;  // No king found (shouldn't happen in a valid position)

    int from_file = FILE(king_sq);
    int from_rank = RANK(king_sq);

    // Try each king move
    for (int i = 0; i < 8; i++) {
        int to_file = from_file + king_dirs[i][0];
        int to_rank = from_rank + king_dirs[i][1];

        // Check if the target square is on the board
        if (to_file < 0 || to_file > 7 || to_rank < 0 || to_rank > 7) {
            continue;
        }

        int to_sq = SQUARE(to_file, to_rank);

        // Only consider captures
        if (board->pieces[to_sq].type != EMPTY && board->pieces[to_sq].color == opponent) {
            add_move(list, king_sq, to_sq, EMPTY, true, false, false, -1);
        }
    }
}
