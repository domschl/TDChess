#include "perft.h"
#include "board.h"
#include "movegen.h"
#include "visualization.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

// Array of PERFT test data
static const PerftData perftData[] = {
    // Reference: https://www.chessprogramming.org/Perft_Results
    {"end games",
     "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     {14, 191, 2812, 43238, 674624, 11030083, 178633661},
     7},

    {"strange bugs",
     "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     {44, 1486, 62379, 2103487, 89941194},
     5},

    {"start pos",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     {20, 400, 8902, 197281, 4865609, 119060324, 3195901860, 84998978956, 2439530234167},
     9},

    {"kiwipete",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     {48, 2039, 97862, 4085603, 193690690, 8031647685},
     6},

    {"position-4",
     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     {6, 264, 9467, 422333, 15833292, 706045033},
     6},

    {"position-4-mirror",
     "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
     {6, 264, 9467, 422333, 15833292, 706045033},
     6},

    {"position-6",
     "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     {46, 2079, 89890, 3894594, 164075551, 6923051137, 287188994746, 11923589843526, 490154852788714},
     9},

    // Reference: https://gist.github.com/peterellisjones/8c46c28141c162d1d8a0f0badbc9cff9
    {"pej-1",
     "r6r/1b2k1bq/8/8/7B/8/8/R3K2R b QK - 3 2",
     {8},
     1},

    {"pej-2",
     "r1bqkbnr/pppppppp/n7/8/8/P7/1PPPPPPP/RNBQKBNR w QqKk - 2 2",
     {19},
     1},

    {"pej-3",
     "r3k2r/p1pp1pb1/bn2Qnp1/2qPN3/1p2P3/2N5/PPPBBPPP/R3K2R b QqKk - 3 2",
     {5},
     1},

    {"pej-4",
     "2kr3r/p1ppqpb1/bn2Qnp1/3PN3/1p2P3/2N5/PPPBBPPP/R3K2R b QK - 3 2",
     {44},
     1},

    {"pej-5",
     "rnb2k1r/pp1Pbppp/2p5/q7/2B5/8/PPPQNnPP/RNB1K2R w QK - 3 9",
     {39},
     1},

    {"pej-6",
     "2r5/3pk3/8/2P5/8/2K5/8/8 w - - 5 4",
     {9},
     1},

    {"pej-7",
     "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     {44, 1486, 62379},
     3},

    {"pej-8",
     "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     {46, 2079, 89890},
     3},

    {"pej-9",
     "3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1",
     {18, 92, 1670, 10138, 185429, 1134888},
     6},

    {"pej-10",
     "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1",
     {13, 102, 1266, 10276, 135655, 1015133},
     6},

    {"pej-11",
     "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1",
     {15, 126, 1928, 13931, 206379, 1440467},
     6},

    {"pej-12",
     "5k2/8/8/8/8/8/8/4K2R w K - 0 1",
     {15, 66, 1198, 6399, 120330, 661072},
     6},

    {"pej-13",
     "3k4/8/8/8/8/8/8/R3K3 w Q - 0 1",
     {16, 71, 1286, 7418, 141077, 803711},
     6},

    {"pej-14",
     "r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1",
     {26, 1141, 27826, 1274206},
     4},

    {"pej-15",
     "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1",
     {44, 1494, 50509, 1720476},
     4},

    {"pej-16",
     "2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1",
     {11, 133, 1442, 19174, 266199, 3821001},
     6},

    {"pej-17",
     "8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1",
     {29, 165, 5160, 31961, 1004658},
     5},

    {"pej-18",
     "4k3/1P6/8/8/8/8/K7/8 w - - 0 1",
     {9, 40, 472, 2661, 38983, 217342},
     6},

    {"pej-19",
     "8/P1k5/K7/8/8/8/8/8 w - - 0 1",
     {6, 27, 273, 1329, 18135, 92683},
     6},

    {"pej-20",
     "K1k5/8/P7/8/8/8/8/8 w - - 0 1",
     {2, 6, 13, 63, 382, 2217},
     6},

    {"pej-21",
     "8/k1P5/8/1K6/8/8/8/8 w - - 0 1",
     {10, 25, 268, 926, 10857, 43261, 567584},
     7},

    {"pej-22",
     "8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1",
     {37, 183, 6559, 23527},
     4}};

static const int NUM_PERFT_TESTS = sizeof(perftData) / sizeof(perftData[0]);

// Add a debug function to the main.c file to help diagnose issues
void debug_board(Board *board) {
    printf("Board state:\n");
    print_board_pretty(board);

    printf("\nBitboard occupancy:\n");
    printf("White: %016llx\n", (unsigned long long)board->occupied[WHITE]);
    printf("Black: %016llx\n", (unsigned long long)board->occupied[BLACK]);
    printf("All pieces: %016llx\n", (unsigned long long)board->all_pieces);

    printf("\nPiece counts:\n");
    int piece_counts[2][7] = {0};
    for (int sq = 0; sq < 64; sq++) {
        Piece p = board->pieces[sq];
        if (p.type != EMPTY) {
            piece_counts[p.color][p.type]++;
        }
    }

    for (int color = 0; color < 2; color++) {
        printf("%s: ", color == WHITE ? "White" : "Black");
        printf("P=%d, N=%d, B=%d, R=%d, Q=%d, K=%d\n",
               piece_counts[color][PAWN], piece_counts[color][KNIGHT],
               piece_counts[color][BISHOP], piece_counts[color][ROOK],
               piece_counts[color][QUEEN], piece_counts[color][KING]);
    }

    MoveList list;
    generate_moves(board, &list);
    printf("\nPseudo-legal moves: %d\n", list.count);

    MoveList legal_list;
    generate_legal_moves(board, &legal_list);
    printf("Legal moves: %d\n", legal_list.count);

    if (legal_list.count > 0) {
        printf("First few legal moves: ");
        for (int i = 0; i < legal_list.count && i < 5; i++) {
            printf("%s ", move_to_string(legal_list.moves[i]));
        }
        printf("\n");
    }
}

// Function to print a detailed perft result for a specific position
void perft_detail(Board *board, int depth) {
    MoveList list;
    generate_legal_moves(board, &list);

    printf("Total legal moves: %d\n", list.count);

    for (int i = 0; i < list.count; i++) {
        Board copy = *board;
        char *move_str = move_to_string(list.moves[i]);

        make_move(&copy, &(list.moves[i]));

        uint64_t nodes = 1;
        if (depth > 1) {
            MoveList next_list;
            generate_legal_moves(&copy, &next_list);
            nodes = next_list.count;

            if (depth > 2) {
                for (int j = 0; j < next_list.count; j++) {
                    Board next_copy = copy;
                    make_move(&next_copy, &(next_list.moves[j]));

                    if (depth > 2) {
                        MoveList next_next_list;
                        generate_legal_moves(&next_copy, &next_next_list);
                        nodes += next_next_list.count;
                    }
                }
            }
        }

        printf("%s: %" PRIu64 "\n", move_str, nodes);
    }
}

uint64_t perft(Board *board, int depth) {
    if (depth == 0) {
        return 1;
    }

    MoveList list;
    generate_legal_moves(board, &list);

    if (depth == 1) {
        return list.count;
    }

    uint64_t nodes = 0;
    for (int i = 0; i < list.count; i++) {
        Board copy = *board;
        make_move(&copy, &(list.moves[i]));

        // Validate board state after making the move
        // validate_board_state(&copy);

        nodes += perft(&copy, depth - 1);
    }

    return nodes;
}

// Function to test move generation performance
void test_perft(int max_depth) {
    Board board;
    setup_default_position(&board);

    printf("Move generation performance test (Perft)\n");
    char fen[100];
    printf("Starting position: %s\n", board_to_fen(&board, fen, sizeof(fen)) ? fen : "Error generating FEN");

    // Debug the board state
    debug_board(&board);

    for (int depth = 1; depth <= max_depth; depth++) {
        clock_t start = clock();
        uint64_t nodes = perft(&board, depth);
        clock_t end = clock();

        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        double nodes_per_second = (time_taken > 0) ? (nodes / time_taken) : 0;

        printf("Depth %d: %" PRIu64 " nodes in %.2f seconds (%.0f nodes/second)\n",
               depth, nodes, time_taken, nodes_per_second);

        // Known perft values for the starting position
        uint64_t known_perft[] = {1, 20, 400, 8902, 197281, 4865609};
        if (depth <= 5) {
            if (nodes == known_perft[depth]) {
                printf("✓ Correct\n");
            } else {
                printf("✗ Incorrect. Expected: %" PRIu64 "\n", known_perft[depth]);
            }
        }
    }
}

// Function to run all PERFT tests
void run_perft_tests(int max_depth) {
    printf("Running PERFT tests (max depth: %d)\n", max_depth);

    int passed = 0;
    int total_tests = 0;

    for (int i = 0; i < NUM_PERFT_TESTS; i++) {
        const PerftData *test = &perftData[i];

        Board board;
        init_board(&board);

        if (!parse_fen(&board, test->fen)) {
            printf("Error: Failed to parse FEN for test '%s'\n", test->name);
            continue;
        }

        printf("\nTest %d: %s\n", i + 1, test->name);

        // Use the smaller of max_depth and test->depth_count
        int depth_to_test = (max_depth < test->depth_count) ? max_depth : test->depth_count;

        bool test_passed = true;

        for (int depth = 1; depth <= depth_to_test; depth++) {
            clock_t start = clock();
            uint64_t nodes = perft(&board, depth);
            clock_t end = clock();

            double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
            double nodes_per_second = (time_taken > 0) ? (nodes / time_taken) : 0;

            unsigned long expected = test->perftcnt[depth - 1];
            total_tests++;

            if (nodes == expected) {
                passed++;
                printf("  Depth %d: %" PRIu64 " nodes (%.0f nodes/second) ✓\n",
                       depth, nodes, nodes_per_second);
            } else {
                test_passed = false;
                printf("  Depth %d: %" PRIu64 " nodes (%.0f nodes/second) ✗ Expected: %lu\n",
                       depth,
                       nodes, nodes_per_second, expected);
            }

            // Don't continue to higher depths if a test fails
            if (!test_passed) {
                break;
            }
        }
    }

    // Print summary
    printf("\nTest Summary: %d/%d tests passed (%.1f%%)\n",
           passed, total_tests, (float)passed / total_tests * 100.0f);
}
