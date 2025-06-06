#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "board.h"
#include "movegen.h"
#include "perft.h"
#include "eval.h"
#include "search.h"
#include "neural.h"
#include "python_binding.h"

// Simple interactive mode to play against the engine
void interactive_mode() {
    Board board;
    setup_default_position(&board);

    printf("TDChess Interactive Mode\n");
    printf("Enter moves in format 'e2e4', 'quit' to exit\n\n");

    char input[10];
    while (1) {
        // Print the board
        print_board(&board);

        // Show available moves
        MoveList list;
        generate_legal_moves(&board, &list);

        printf("Available moves (%d):\n", list.count);
        for (int i = 0; i < list.count; i++) {
            printf("%s ", move_to_string(list.moves[i]));
            if ((i + 1) % 10 == 0) printf("\n");
        }
        printf("\n");

        // Get user input
        printf("Enter move: ");
        scanf("%s", input);

        if (strcmp(input, "quit") == 0) {
            break;
        }

        // Parse the move
        Move move = string_to_move(&board, input);

        // Validate the move
        bool valid_move = false;
        for (int i = 0; i < list.count; i++) {
            if (move.from == list.moves[i].from &&
                move.to == list.moves[i].to &&
                move.promotion == list.moves[i].promotion) {

                valid_move = true;
                move = list.moves[i];  // Use the fully populated move
                break;
            }
        }

        if (valid_move) {
            make_move(&board, &move);
        } else {
            printf("Invalid move! Try again.\n");
        }
    }
}

// Test the evaluation function
void test_evaluation(void) {
    Board board;
    setup_default_position(&board);

    printf("Evaluating starting position:\n");
    print_board(&board);

    float eval = evaluate_basic(&board);
    printf("Evaluation: %.2f\n", eval);

    // Make a move to test different positions
    Move e4 = {SQUARE(SQUARE_FILE_E, SQUARE_RANK_2), SQUARE(SQUARE_FILE_E, SQUARE_RANK_4), EMPTY, false, false, false, -1};
    make_move(&board, &e4);

    printf("\nAfter 1. e4:\n");
    print_board(&board);

    eval = evaluate_basic(&board);
    printf("Evaluation: %.2f\n", eval);
}

// Add this function to play against the computer
void play_against_computer(int depth) {
    Board board;
    setup_default_position(&board);

    printf("Playing against computer (depth: %d)\n", depth);
    printf("Enter moves in algebraic notation (e.g., e2e4)\n");
    printf("Enter 'quit' to exit\n\n");

    while (1) {
        print_board(&board);
        printf("\nEvaluation: %.2f\n", evaluate_basic(&board));

        // Generate legal moves
        MoveList moves;
        generate_legal_moves(&board, &moves);

        if (moves.count == 0) {
            // Check if in check (checkmate)
            int king_square = -1;
            for (int sq = 0; sq < 64; sq++) {
                if (board.pieces[sq].type == KING && board.pieces[sq].color == board.side_to_move) {
                    king_square = sq;
                    break;
                }
            }

            if (king_square != -1 && is_square_attacked(&board, king_square, !board.side_to_move)) {
                printf("Checkmate! %s wins.\n", board.side_to_move == WHITE ? "Black" : "White");
            } else {
                printf("Stalemate! Game is a draw.\n");
            }
            break;
        }

        if (board.side_to_move == WHITE) {
            printf("Your move (white): ");
            char input[10];
            if (scanf("%9s", input) != 1) {
                printf("Error reading input\n");
                break;
            }

            if (strcmp(input, "quit") == 0) {
                break;
            }

            // Parse the move
            if (strlen(input) < 4) {
                printf("Invalid move format. Use e2e4 format.\n");
                continue;
            }

            int from_file = input[0] - 'a';
            int from_rank = input[1] - '1';
            int to_file = input[2] - 'a';
            int to_rank = input[3] - '1';

            if (from_file < 0 || from_file > 7 || from_rank < 0 || from_rank > 7 ||
                to_file < 0 || to_file > 7 || to_rank < 0 || to_rank > 7) {
                printf("Invalid move coordinates.\n");
                continue;
            }

            int from_sq = SQUARE(from_file, from_rank);
            int to_sq = SQUARE(to_file, to_rank);

            // Find matching legal move
            bool move_found = false;
            for (int i = 0; i < moves.count; i++) {
                if (moves.moves[i].from == from_sq && moves.moves[i].to == to_sq) {
                    // Handle promotion
                    if (board.pieces[from_sq].type == PAWN && to_rank == 7) {
                        moves.moves[i].promotion = QUEEN;  // Default to queen promotion
                    }
                    make_move(&board, &(moves.moves[i]));
                    move_found = true;
                    break;
                }
            }

            if (!move_found) {
                printf("Illegal move. Try again.\n");
                continue;
            }
        } else {
            // Computer's move
            Move computer_move = get_computer_move(&board, depth);
            make_move(&board, &computer_move);
        }
    }
}

// Test the neural input representation
void test_neural_input(void) {
    Board board;
    setup_default_position(&board);

    printf("Testing neural input representation for starting position:\n");
    print_board(&board);
    print_tensor_representation(&board);

    // Test with a more complex position
    if (!parse_fen(&board, "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")) {
        printf("Failed to parse FEN\n");
        return;
    }

    printf("\nTesting neural input representation for position after 1.e4 e5 2.Nf3 Nc6:\n");
    print_board(&board);
    print_tensor_representation(&board);
}

// Add a function to test neural evaluation

void test_neural_evaluation(const char *model_path) {
    if (!is_neural_available()) {
        printf("Neural network support is not available\n");
        return;
    }

    // Load the neural evaluator
    NeuralEvaluator *evaluator = load_neural_evaluator(model_path);
    if (!evaluator) {
        printf("Failed to load neural model from %s\n", model_path);
        return;
    }

    // Set as the global evaluator
    set_neural_evaluator(evaluator);

    // Create a board with the starting position
    Board board;
    setup_default_position(&board);

    // Print the board
    printf("Evaluating starting position:\n");
    print_board(&board);

    // Evaluate using the neural network
    float score = neural_evaluate_position(evaluator, &board);
    printf("Neural evaluation: %.3f\n", score);

    // Compare with classical evaluation
    float classic_score = evaluate_basic(&board);
    printf("Classical evaluation: %.3f\n", classic_score);

    // Try a few common opening positions
    const char *test_positions[] = {
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",    // After 1.e4
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  // After 1.e4 e5
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"  // After 1.e4 e5 2.Nf3
    };

    for (int i = 0; i < 3; i++) {
        printf("\nEvaluating position %d:\n", i + 1);

        if (!parse_fen(&board, test_positions[i])) {
            printf("Failed to parse FEN: %s\n", test_positions[i]);
            continue;
        }

        print_board(&board);

        // Evaluate using the neural network
        float score = neural_evaluate_position(evaluator, &board);
        printf("Neural evaluation: %.3f\n", score);

        // Compare with classical evaluation
        float classic_score = evaluate_basic(&board);
        printf("Classical evaluation: %.3f\n", classic_score);
    }

    // Clean up
    free_neural_evaluator(evaluator);
    set_neural_evaluator(NULL);
}

// Generate a dataset of positions for neural network training
void generate_training_dataset(const char *filename, int num_positions, int search_depth) {
    printf("Generating training dataset with %d positions...\n", num_positions);

    // Allocate memory for positions and evaluations
    Board *positions = (Board *)malloc(num_positions * sizeof(Board));
    float *evaluations = (float *)malloc(num_positions * sizeof(float));

    if (!positions || !evaluations) {
        printf("Failed to allocate memory for dataset\n");
        if (positions) free(positions);
        if (evaluations) free(evaluations);
        return;
    }

    // Start with the default position
    Board current_board;
    setup_default_position(&current_board);

    // Use alpha-beta search to play some moves and collect positions
    int pos_count = 0;

    // Collect first position
    positions[pos_count] = current_board;
    evaluations[pos_count] = evaluate_basic(&current_board);
    pos_count++;

    // Play moves and collect positions
    uint64_t nodes = 0;
    while (pos_count < num_positions) {
        // Search for best move
        Move best_move;
        find_best_move(&current_board, search_depth, &best_move, &nodes);

        // Make the move
        make_move(&current_board, &best_move);

        // Collect position
        positions[pos_count] = current_board;
        evaluations[pos_count] = evaluate_basic(&current_board);
        pos_count++;

        // Print progress
        if (pos_count % 10 == 0) {
            printf("Generated %d/%d positions\n", pos_count, num_positions);
        }

        // If game is over, start a new game
        MoveList moves;
        generate_legal_moves(&current_board, &moves);
        if (moves.count == 0 || current_board.halfmove_clock >= 100) {
            setup_default_position(&current_board);
        }
    }

    // Export positions to dataset
    if (!export_positions_to_dataset(filename, positions, evaluations, num_positions)) {
        printf("Failed to export positions to dataset\n");
    } else {
        printf("Successfully exported %d positions to %s\n", num_positions, filename);
    }

    // Free memory
    free(positions);
    free(evaluations);
}

// Update the main function to handle the test-mode option
int main(int argc, char **argv) {
    printf("TDChess - A chess engine\n\n");

    if (argc > 1) {
        // Process command-line arguments
        if (strcmp(argv[1], "perft") == 0) {
            int depth = (argc > 2) ? atoi(argv[2]) : 5;
            test_perft(depth);
        } else if (strcmp(argv[1], "perft-detail") == 0) {
            Board board;
            setup_default_position(&board);

            int depth = (argc > 2) ? atoi(argv[2]) : 1;
            perft_detail(&board, depth);
        } else if (strcmp(argv[1], "test") == 0) {
            // New test-mode option
            int max_depth = (argc > 2) ? atoi(argv[2]) : 3;
            run_perft_tests(max_depth);
        } else if (strcmp(argv[1], "eval") == 0) {
            // Test evaluation function
            test_evaluation();
        } else if (strcmp(argv[1], "play") == 0) {
            // Play against computer
            int depth = (argc > 2) ? atoi(argv[2]) : 3;
            play_against_computer(depth);
        } else if (strcmp(argv[1], "neural") == 0) {
            // Test neural input representation
            test_neural_input();
        } else if (strcmp(argv[1], "neural-eval") == 0) {
            // Test neural evaluation
            const char *model_path = (argc > 2) ? argv[2] : "chess_model.onnx";
            test_neural_evaluation(model_path);
        } else if (strcmp(argv[1], "generate-dataset") == 0) {
            // Generate training dataset
            const char *filename = (argc > 2) ? argv[2] : "chess_dataset.json";
            int num_positions = (argc > 3) ? atoi(argv[3]) : 1000;
            int search_depth = (argc > 4) ? atoi(argv[4]) : 3;
            generate_training_dataset(filename, num_positions, search_depth);
        } else {
            printf("Unknown command: %s\n", argv[1]);
            printf("Available commands:\n");
            printf("  perft [depth]       - Run perft to specified depth\n");
            printf("  perft-detail [depth] - Show detailed perft results\n");
            printf("  test [max_depth]     - Run perft tests on standard positions\n");
            printf("  eval                 - Test evaluation function\n");
            printf("  play [depth]         - Play against computer\n");
            printf("  neural              - Test neural input representation\n");
            printf("  neural-eval [model]  - Test neural evaluation with ONNX model\n");
            printf("  generate-dataset [file] [count] [depth] - Generate training dataset\n");
        }
    } else {
        // Default to interactive mode
        interactive_mode();
    }

    return 0;
}
