#include "movegen.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

// Initialize a move list
static void init_move_list(MoveList *list) {
    list->count = 0;
}

// Update the add_move function to initialize new fields

static void add_move(MoveList *list, int from, int to, PieceType promotion, bool capture, bool castling, bool en_passant, int captured_square) {
    // Check if the list is full
    if (list->count >= MAX_MOVES) {
        return;
    }

    // Initialize the move
    Move move;
    move.from = from;
    move.to = to;
    move.promotion = promotion;
    move.capture = capture;
    move.castling = castling;
    move.en_passant = en_passant;
    move.captured_piece_square = captured_square;

    // Initialize state preservation fields with default values
    // These will be properly set by make_move
    move.captured_piece_type = EMPTY;
    move.captured_piece_color = WHITE;  // Default value, will be overwritten if capture
    move.old_castle_rights = 0;
    move.old_en_passant = -1;
    move.old_halfmove_clock = 0;

    // Add the move to the list
    list->moves[list->count] = move;
    list->scores[list->count] = 0;  // Initialize move score
    list->count++;
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
    make_move(&temp_board, &move);

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
    int start_rank = (side == WHITE) ? SQUARE_RANK_2 : SQUARE_RANK_7;
    int promotion_rank = (side == WHITE) ? SQUARE_RANK_7 : SQUARE_RANK_2;

    // Get all pawns of the current side
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type != PAWN || board->pieces[sq].color != side) {
            continue;
        }

        int file = SQUARE_FILE(sq);
        int rank = SQUARE_RANK(sq);

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
                int from_file = SQUARE_FILE(square);
                int from_rank = SQUARE_RANK(square);
                int to_file = SQUARE_FILE(to);
                int to_rank = SQUARE_RANK(to);

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
            int from_file = SQUARE_FILE(square);
            int from_rank = SQUARE_RANK(square);

            for (int i = 0; i < 4; i++) {
                int dir = bishop_dirs[i];
                int to = square;
                int current_file = from_file;
                int current_rank = from_rank;

                while (true) {
                    to += dir;

                    // Check if the target square is on the board
                    if (to < 0 || to >= 64) break;

                    int to_file = SQUARE_FILE(to);
                    int to_rank = SQUARE_RANK(to);

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
            int from_file = SQUARE_FILE(square);
            int from_rank = SQUARE_RANK(square);

            for (int i = 0; i < 4; i++) {
                int dir = rook_dirs[i];
                int to = square;
                int current_file = from_file;
                int current_rank = from_rank;

                while (true) {
                    to += dir;

                    // Check if the target square is on the board
                    if (to < 0 || to >= 64) break;

                    int to_file = SQUARE_FILE(to);
                    int to_rank = SQUARE_RANK(to);

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
            int from_file = SQUARE_FILE(square);
            int from_rank = SQUARE_RANK(square);

            for (int i = 0; i < 8; i++) {
                int dir = queen_dirs[i];
                int to = square;
                int current_file = from_file;
                int current_rank = from_rank;

                while (true) {
                    to += dir;

                    // Check if the target square is on the board
                    if (to < 0 || to >= 64) break;

                    int to_file = SQUARE_FILE(to);
                    int to_rank = SQUARE_RANK(to);

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
                int from_file = SQUARE_FILE(square);
                int from_rank = SQUARE_RANK(square);
                int to_file = SQUARE_FILE(to);
                int to_rank = SQUARE_RANK(to);

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
                    board->pieces[SQUARE(SQUARE_FILE_F, SQUARE_RANK_1)].type == EMPTY &&
                    board->pieces[SQUARE(SQUARE_FILE_G, SQUARE_RANK_1)].type == EMPTY) {

                    // Check if king and path to castling are not in check
                    if (!is_square_attacked(board, square, BLACK) &&
                        !is_square_attacked(board, SQUARE(SQUARE_FILE_F, SQUARE_RANK_1), BLACK)) {
                        add_move(list, square, SQUARE(SQUARE_FILE_G, SQUARE_RANK_1), EMPTY, false, true, false, -1);
                    }
                }

                // White queenside castling
                if ((board->castle_rights & CASTLE_WHITE_QUEENSIDE) &&
                    board->pieces[SQUARE(SQUARE_FILE_D, SQUARE_RANK_1)].type == EMPTY &&
                    board->pieces[SQUARE(SQUARE_FILE_C, SQUARE_RANK_1)].type == EMPTY &&
                    board->pieces[SQUARE(SQUARE_FILE_B, SQUARE_RANK_1)].type == EMPTY) {

                    // Check if king and path to castling are not in check
                    if (!is_square_attacked(board, square, BLACK) &&
                        !is_square_attacked(board, SQUARE(SQUARE_FILE_D, SQUARE_RANK_1), BLACK)) {
                        add_move(list, square, SQUARE(SQUARE_FILE_C, SQUARE_RANK_1), EMPTY, false, true, false, -1);
                    }
                }
            } else {
                // Black kingside castling
                if ((board->castle_rights & CASTLE_BLACK_KINGSIDE) &&
                    board->pieces[SQUARE(SQUARE_FILE_F, SQUARE_RANK_8)].type == EMPTY &&
                    board->pieces[SQUARE(SQUARE_FILE_G, SQUARE_RANK_8)].type == EMPTY) {

                    // Check if king and path to castling are not in check
                    if (!is_square_attacked(board, square, WHITE) &&
                        !is_square_attacked(board, SQUARE(SQUARE_FILE_F, SQUARE_RANK_8), WHITE)) {
                        add_move(list, square, SQUARE(SQUARE_FILE_G, SQUARE_RANK_8), EMPTY, false, true, false, -1);
                    }
                }

                // Black queenside castling
                if ((board->castle_rights & CASTLE_BLACK_QUEENSIDE) &&
                    board->pieces[SQUARE(SQUARE_FILE_D, SQUARE_RANK_8)].type == EMPTY &&
                    board->pieces[SQUARE(SQUARE_FILE_C, SQUARE_RANK_8)].type == EMPTY &&
                    board->pieces[SQUARE(SQUARE_FILE_B, SQUARE_RANK_8)].type == EMPTY) {

                    // Check if king and path to castling are not in check
                    if (!is_square_attacked(board, square, WHITE) &&
                        !is_square_attacked(board, SQUARE(SQUARE_FILE_D, SQUARE_RANK_8), WHITE)) {
                        add_move(list, square, SQUARE(SQUARE_FILE_C, SQUARE_RANK_8), EMPTY, false, true, false, -1);
                    }
                }
            }

            break;  // We found the king, no need to continue
        }
    }
}

/**
 * Updates all bitboards to match the piece array
 * This ensures consistency between the two board representations
 *
 * @param board The board structure
 */
static void update_bitboards(Board *board) {
    // Clear all bitboards
    for (int color = WHITE; color <= BLACK; color++) {
        board->occupied[color] = 0;
        for (int piece_type = PAWN; piece_type <= KING; piece_type++) {
            board->piece_bb[color][piece_type] = 0;
        }
    }
    board->all_pieces = 0;

    // Populate bitboards based on piece array
    for (int sq = 0; sq < 64; sq++) {
        Piece piece = board->pieces[sq];
        if (piece.type != EMPTY) {
            Bitboard bb = square_to_bitboard(sq);

            // Update piece-specific bitboard
            board->piece_bb[piece.color][piece.type] |= bb;

            // Update color bitboard
            board->occupied[piece.color] |= bb;

            // Update combined bitboard
            board->all_pieces |= bb;
        }
    }
}

// Make a move on the board
bool make_move(Board *board, Move *move) {
    // Store state information for unmake_move
    move->old_castle_rights = board->castle_rights;
    move->old_en_passant = board->en_passant_square;
    move->old_halfmove_clock = board->halfmove_clock;

    // Get the moving piece
    Piece moving_piece = board->pieces[move->from];

    // Store information about any captured piece
    if (move->capture) {
        if (move->en_passant) {
            // For en passant, the captured piece is a pawn on a different square
            // The actual square of the captured pawn is move->captured_piece_square
            move->captured_piece_type = PAWN;  // The piece type is always PAWN
            move->captured_piece_color = (board->side_to_move == WHITE) ? BLACK : WHITE;

            // Clear the captured pawn from the board
            board->pieces[move->captured_piece_square].type = EMPTY;
        } else {
            // Normal capture - store the captured piece information
            move->captured_piece_type = board->pieces[move->to].type;
            move->captured_piece_color = board->pieces[move->to].color;
        }
    } else {
        // No capture
        move->captured_piece_type = EMPTY;
    }

    // Move the piece on the board
    board->pieces[move->to] = moving_piece;
    board->pieces[move->from].type = EMPTY;

    // Update king position if a king moved
    if (moving_piece.type == KING) {
        board->king_pos[moving_piece.color] = move->to;
    }

    // Handle promotion
    if (move->promotion != EMPTY) {
        board->pieces[move->to].type = move->promotion;
    }

    // Handle castling
    if (move->castling) {
        // King has already been moved to move->to and king_pos updated.
        // Now move the rook.
        int rook_from = 0, rook_to = 0;

        if (move->to == G1) {  // White kingside (king moved E1->G1)
            rook_from = H1;
            rook_to = F1;
        } else if (move->to == C1) {  // White queenside (king moved E1->C1)
            rook_from = A1;
            rook_to = D1;
        } else if (move->to == G8) {  // Black kingside (king moved E8->G8)
            rook_from = H8;
            rook_to = F8;
        } else if (move->to == C8) {  // Black queenside (king moved E8->C8)
            rook_from = A8;
            rook_to = D8;
        }

        // Move the rook
        board->pieces[rook_to] = board->pieces[rook_from];
        board->pieces[rook_from].type = EMPTY;
    }

    // Update castling rights
    // This needs to consider the original position of the piece (move->from)
    // if it was a king or rook, or the destination (move->to) if a rook was captured there.
    // int original_castle_rights = board->castle_rights;

    if (moving_piece.type == KING) {
        if (moving_piece.color == WHITE) {
            board->castle_rights &= ~(CASTLE_WHITE_KINGSIDE | CASTLE_WHITE_QUEENSIDE);
        } else {
            board->castle_rights &= ~(CASTLE_BLACK_KINGSIDE | CASTLE_BLACK_QUEENSIDE);
        }
    }
    // Check if a rook moved from its starting square
    if (move->from == H1) board->castle_rights &= ~CASTLE_WHITE_KINGSIDE;
    if (move->from == A1) board->castle_rights &= ~CASTLE_WHITE_QUEENSIDE;
    if (move->from == H8) board->castle_rights &= ~CASTLE_BLACK_KINGSIDE;
    if (move->from == A8) board->castle_rights &= ~CASTLE_BLACK_QUEENSIDE;

    // Check if a rook was captured on its starting square
    if (move->capture && !move->en_passant) {  // en_passant capture is handled separately for piece type
        if (move->to == H1 && move->captured_piece_type == ROOK && move->captured_piece_color == WHITE) board->castle_rights &= ~CASTLE_WHITE_KINGSIDE;
        if (move->to == A1 && move->captured_piece_type == ROOK && move->captured_piece_color == WHITE) board->castle_rights &= ~CASTLE_WHITE_QUEENSIDE;
        if (move->to == H8 && move->captured_piece_type == ROOK && move->captured_piece_color == BLACK) board->castle_rights &= ~CASTLE_BLACK_KINGSIDE;
        if (move->to == A8 && move->captured_piece_type == ROOK && move->captured_piece_color == BLACK) board->castle_rights &= ~CASTLE_BLACK_QUEENSIDE;
    }

    // Update the en passant square
    if (moving_piece.type == PAWN && abs(move->to - move->from) == 16) {
        // Double pawn push - set en passant square
        board->en_passant_square = (move->from + move->to) / 2;
    } else {
        // Clear en passant square
        board->en_passant_square = -1;
    }

    // Update halfmove clock
    if (moving_piece.type == PAWN || move->capture) {
        // Pawn move or capture - reset halfmove clock
        board->halfmove_clock = 0;
    } else {
        // Increment halfmove clock
        board->halfmove_clock++;
    }

    // Update fullmove number
    if (board->side_to_move == BLACK) {
        board->fullmove_number++;
    }

    // Switch side to move
    board->side_to_move = (board->side_to_move == WHITE) ? BLACK : WHITE;

    // Update bitboards
    update_bitboards(board);

    return true;
}

// Convert move to algebraic notation
char *move_to_string(Move move) {
    static char str[6];

    char files[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
    char ranks[8] = {'1', '2', '3', '4', '5', '6', '7', '8'};

    int from_file = SQUARE_FILE(move.from);
    int from_rank = SQUARE_RANK(move.from);
    int to_file = SQUARE_FILE(move.to);
    int to_rank = SQUARE_RANK(move.to);

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
    int promotion_rank = (side == WHITE) ? SQUARE_RANK_7 : SQUARE_RANK_2;

    // Get all pawns of the current side
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type != PAWN || board->pieces[sq].color != side) {
            continue;
        }

        int file = SQUARE_FILE(sq);
        int rank = SQUARE_RANK(sq);

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

        int from_file = SQUARE_FILE(sq);
        int from_rank = SQUARE_RANK(sq);

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
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

    // Get all bishops of the current side
    for (int sq = 0; sq < 64; sq++) {
        if (board->pieces[sq].type != BISHOP || board->pieces[sq].color != side) {
            continue;
        }

        int from_file = SQUARE_FILE(sq);
        int from_rank = SQUARE_RANK(sq);

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

        int from_file = SQUARE_FILE(sq);
        int from_rank = SQUARE_RANK(sq);

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

        int from_file = SQUARE_FILE(sq);
        int from_rank = SQUARE_RANK(sq);

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

    int from_file = SQUARE_FILE(king_sq);
    int from_rank = SQUARE_RANK(king_sq);

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

// Correct implementation of unmake_move

void unmake_move(Board *board, Move move) {
    // Switch side to move back
    board->side_to_move = (board->side_to_move == WHITE) ? BLACK : WHITE;

    // If it was black's move (now white to move after switching back), decrement the full move counter
    if (board->side_to_move == WHITE) {
        board->fullmove_number--;
    }

    // Get the piece that was moved (it's currently at move.to)
    Piece moved_piece = board->pieces[move.to];

    // Handle promotion - change piece type back to PAWN before moving it
    if (move.promotion != EMPTY) {
        moved_piece.type = PAWN;
    }

    // Move the piece back to its original square
    board->pieces[move.from] = moved_piece;
    // The destination square (move.to) will be correctly set below (either empty or restored captured piece)

    // Update king position if a king was moved
    if (moved_piece.type == KING) {
        board->king_pos[moved_piece.color] = move.from;
    }

    // Handle castling (king and rook movements)
    if (move.castling) {
        // King has been moved back to move.from and king_pos updated.
        // Now move the rook back.
        int rook_original_from = 0, rook_original_to = 0;  // original_from is where rook is now, original_to is where it came from

        if (move.to == G1) {  // White kingside (king was E1->G1, rook H1->F1)
            rook_original_from = F1;
            rook_original_to = H1;
        } else if (move.to == C1) {  // White queenside (king E1->C1, rook A1->D1)
            rook_original_from = D1;
            rook_original_to = A1;
        } else if (move.to == G8) {  // Black kingside (king E8->G8, rook H8->F8)
            rook_original_from = F8;
            rook_original_to = H8;
        } else if (move.to == C8) {  // Black queenside (king E8->C8, rook A8->D8)
            rook_original_from = D8;
            rook_original_to = A8;
        }

        // Move the rook back
        board->pieces[rook_original_to] = board->pieces[rook_original_from];
        board->pieces[rook_original_from].type = EMPTY;
        // Clear the square the king moved to during castling (it's now empty or will be set by captured piece logic if it was a capture, though castling isn't a capture)
        board->pieces[move.to].type = EMPTY;

    } else {  // Non-castling moves
        // Restore captured piece or clear destination
        if (move.capture) {
            if (move.en_passant) {
                // For en passant, restore the captured pawn
                // The square move.captured_piece_square holds the actual pawn that was captured
                board->pieces[move.captured_piece_square].type = PAWN;  // It's always a pawn
                board->pieces[move.captured_piece_square].color = move.captured_piece_color;

                // The square the capturing pawn moved to (move.to) should become empty
                board->pieces[move.to].type = EMPTY;
            } else {
                // Normal capture - restore the captured piece at move.to
                board->pieces[move.to].type = move.captured_piece_type;
                board->pieces[move.to].color = move.captured_piece_color;
            }
        } else {
            // No capture - the destination square (move.to) becomes empty
            board->pieces[move.to].type = EMPTY;
        }
    }

    // Restore preserved state
    board->castle_rights = move.old_castle_rights;
    board->en_passant_square = move.old_en_passant;
    board->halfmove_clock = move.old_halfmove_clock;

    // Update bitboards
    update_bitboards(board);
}
