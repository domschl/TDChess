#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include "board.h"
#include "movegen.h"
#include "perft.h"
#include "eval.h"
#include "search.h"
#include "neural.h"
#include "python_binding.h"
#include "td_learning.h"
#include "visualization.h"
#include "self_play.h"
#include "pytorch_binding.h"  // Add this to your imports

// Add this function declaration at the top of the file with other function declarations

// Function to generate training dataset for neural network
void generate_training_dataset(const char *filename, int num_positions, int search_depth);

// Add this implementation somewhere in the file, before it's called

/**
 * Generate a training dataset for neural network training
 *
 * @param filename The name of the file to save the dataset to
 * @param num_positions The number of positions to generate
 * @param search_depth The search depth to use for move selection
 */
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
    memcpy(&positions[pos_count], &current_board, sizeof(Board));
    // evaluate_position returns pawn units from current_board.side_to_move's perspective.
    // For starting position, side_to_move is WHITE.
    float eval_pawn_units_initial = evaluate_position(&current_board);
    evaluations[pos_count] = eval_pawn_units_initial * 100.0f;  // Convert to centipawns
    pos_count++;

    // Play moves and collect positions
    uint64_t nodes = 0;
    while (pos_count < num_positions) {
        // Search for best move
        Move best_move;
        // score is V(s_t+1) from P_t's perspective (player at s_t), in pawn units
        float score_pawn_units = find_best_move(&current_board, search_depth, &best_move, &nodes, 0);

        // Make the move
        make_move(&current_board, &best_move);  // current_board is now s_t+1, side_to_move is P_t+1

        // Collect position
        memcpy(&positions[pos_count], &current_board, sizeof(Board));

        // eval_s_t_plus_1_curr_player_pawn_units is V(s_t+1) from P_t+1's perspective, in pawn units
        float eval_s_t_plus_1_curr_player_pawn_units = -score_pawn_units;

        float eval_s_t_plus_1_white_perspective_pawn_units;
        if (current_board.side_to_move == WHITE) {  // P_t+1 is White
            eval_s_t_plus_1_white_perspective_pawn_units = eval_s_t_plus_1_curr_player_pawn_units;
        } else {  // P_t+1 is Black, so negate to get White's perspective
            eval_s_t_plus_1_white_perspective_pawn_units = -eval_s_t_plus_1_curr_player_pawn_units;
        }

        evaluations[pos_count] = eval_s_t_plus_1_white_perspective_pawn_units * 100.0f;  // Convert to centipawns
        pos_count++;

        // Print progress
        if (pos_count % 100 == 0) {  // Adjusted progress printing
            printf("Generated %d/%d positions\n", pos_count, num_positions);
        }

        // If game is over, start a new game
        MoveList moves;
        generate_legal_moves(&current_board, &moves);
        if (moves.count == 0 || current_board.halfmove_clock >= 100) {
            setup_default_position(&current_board);
            // Optionally, add the new starting position to the dataset if not full
            if (pos_count < num_positions) {
                memcpy(&positions[pos_count], &current_board, sizeof(Board));
                float eval_pawn_units_reset = evaluate_position(&current_board);  // From White's perspective
                evaluations[pos_count] = eval_pawn_units_reset * 100.0f;
                pos_count++;
            }
        }
    }

    // Export positions to dataset
    if (!export_positions_to_dataset(filename, positions, evaluations, pos_count)) {  // Use pos_count
        printf("Failed to export positions to dataset\n");
    } else {
        printf("Successfully exported %d positions to %s\n", pos_count, filename);  // Use pos_count
    }

    // Free memory
    free(positions);
    free(evaluations);
}

// Simple interactive mode to play against the engine
void interactive_mode() {
    Board board;
    setup_default_position(&board);

    printf("TDChess Interactive Mode\n");
    printf("Enter moves in format 'e2e4', 'quit' to exit\n\n");

    char input[10];
    while (1) {
        // Print the board
        print_board_pretty(&board);

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
    print_board_pretty(&board);

    float eval = evaluate_basic(&board);
    printf("Evaluation: %.2f\n", eval);

    // Make a move to test different positions
    Move e4 = {
        .from = SQUARE(SQUARE_FILE_E, SQUARE_RANK_2),
        .to = SQUARE(SQUARE_FILE_E, SQUARE_RANK_4),
        .promotion = EMPTY,
        .capture = false,
        .castling = false,
        .en_passant = false,
        .captured_piece_square = -1,
        .captured_piece_type = EMPTY,
        .captured_piece_color = WHITE,  // Default, doesn't matter for non-captures
        .old_castle_rights = 0,         // Will be set by make_move
        .old_en_passant = -1,           // Will be set by make_move
        .old_halfmove_clock = 0         // Will be set by make_move
    };
    make_move(&board, &e4);

    printf("\nAfter 1. e4:\n");
    print_board_pretty(&board);

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
        print_board_pretty(&board);
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
            uint64_t nodes = 0;
            int search_depth = depth;     // Use the provided search depth
            Board current_board = board;  // Copy current board for search
            Move computer_move;
            float score_pawn_units = find_best_move(&current_board, search_depth, &computer_move, &nodes, 1);
            printf("Computer move: %s (score: %.2f, nodes: %" PRIu64 ")\n",
                   move_to_string(computer_move), score_pawn_units, nodes);

            // Add debug print to track the score
            printf("DEBUG: Score after find_best_move: %.8f\n", score_pawn_units);

            make_move(&board, &computer_move);
        }
    }
}

/**
 * Play against the computer using neural network evaluation
 *
 * @param model_path Path to the neural model file
 * @param depth Search depth for the computer
 */
void play_with_neural(const char *model_path, int depth) {
    printf("Playing with neural network evaluation (depth: %d)\n", depth);
    printf("Loading neural model: %s\n", model_path);

    // Initialize the neural network
    if (!initialize_neural(model_path)) {
        printf("Failed to initialize neural model from %s\n", model_path);
        return;
    }

    // Ensure neural model is unloaded when function exits
    atexit(shutdown_neural);

    // Store original evaluation type
    EvaluationType original_type = get_evaluation_type();

    // Switch to neural evaluation mode
    set_evaluation_type(EVAL_NEURAL);

    // Use the existing play function
    play_against_computer(depth);

    // Reset to original evaluation type
    set_evaluation_type(original_type);
}

// Test the neural input representation
void cmd_neural_input(void) {
    printf("Testing neural input representation:\n");
    test_neural_input();  // Call the function defined in neural.h
}

// Replace the conflicting test_neural_evaluation function with a wrapper
// that uses the correct implementation from neural.h
void cmd_test_neural_model(const char *model_path) {
    // Initialize neural evaluator
    if (!initialize_neural(model_path)) {
        printf("Failed to initialize neural evaluator with model: %s\n", model_path);
        return;
    }

    // Make sure to clean up when done
    atexit(shutdown_neural);

    // Test positions
    Board board;

    // Position 1: Starting position
    setup_default_position(&board);
    printf("Evaluating starting position:\n");
    print_board_pretty(&board);
    test_neural_evaluation(&board);  // Use the function from neural.h
    printf("\n");

    // Position 2: After 1.e4
    setup_default_position(&board);
    Move e4 = {
        .from = SQUARE(SQUARE_FILE_E, SQUARE_RANK_2),
        .to = SQUARE(SQUARE_FILE_E, SQUARE_RANK_4),
        .promotion = EMPTY,
        .capture = false,
        .castling = false,
        .en_passant = false,
        .captured_piece_square = -1,
        .captured_piece_type = EMPTY,
        .captured_piece_color = WHITE,
        .old_castle_rights = 0,
        .old_en_passant = -1,
        .old_halfmove_clock = 0};
    make_move(&board, &e4);
    printf("Evaluating position 1:\n");
    print_board_pretty(&board);
    test_neural_evaluation(&board);
    printf("\n");

    // Position 3: After 1.e4 e5
    Move e5 = {
        .from = SQUARE(SQUARE_FILE_E, SQUARE_RANK_7),
        .to = SQUARE(SQUARE_FILE_E, SQUARE_RANK_5),
        .promotion = EMPTY,
        .capture = false,
        .castling = false,
        .en_passant = false,
        .captured_piece_square = -1,
        .captured_piece_type = EMPTY,
        .captured_piece_color = BLACK,
        .old_castle_rights = 0,
        .old_en_passant = -1,
        .old_halfmove_clock = 0};
    make_move(&board, &e5);
    printf("Evaluating position 2:\n");
    print_board_pretty(&board);
    test_neural_evaluation(&board);
    printf("\n");

    // Position 4: After 2.Nf3
    Move nf3 = {
        .from = SQUARE(SQUARE_FILE_G, SQUARE_RANK_1),
        .to = SQUARE(SQUARE_FILE_F, SQUARE_RANK_3),
        .promotion = EMPTY,
        .capture = false,
        .castling = false,
        .en_passant = false,
        .captured_piece_square = -1,
        .captured_piece_type = EMPTY,
        .captured_piece_color = WHITE,
        .old_castle_rights = 0,
        .old_en_passant = -1,
        .old_halfmove_clock = 0};
    make_move(&board, &nf3);
    printf("Evaluating position 3:\n");
    print_board_pretty(&board);
    test_neural_evaluation(&board);
    printf("\n");

    // Explicit cleanup is not needed due to atexit() registration
}

// Add new td-lambda command
void cmd_td_lambda_training(const char *initial_model, const char *output_model,
                            int num_games, float lambda, float temperature);

// Update the function implementation with temperature
void cmd_td_lambda_training(const char *initial_model, const char *output_model,
                            int num_games, float lambda, float temperature) {
    printf("Starting TD-Lambda training cycle\n");

    // Set up parameters
    TDLambdaParams params;
    params.lambda = lambda;
    params.learning_rate = 0.0001f;
    params.temperature = temperature;  // Use the temperature parameter
    params.num_games = num_games;
    params.max_moves = 100;
    params.model_path = initial_model;

    // Create dataset filename
    char dataset_path[256];
    snprintf(dataset_path, sizeof(dataset_path), "%s.dataset.json", output_model);
    params.output_path = dataset_path;

    // Generate TD-Lambda dataset
    if (!generate_td_lambda_dataset(&params)) {
        printf("Failed to generate TD-Lambda dataset\n");
        return;
    }

    // Train the model using Python script
    char command[512];
    snprintf(command, sizeof(command),
             "python train_neural.py --dataset %s --output %s --epochs 100 --batch-size 128 --learning-rate %.5f",
             dataset_path, output_model, params.learning_rate);

    printf("Running training command: %s\n", command);
    int result = system(command);

    if (result != 0) {
        printf("Training failed with exit code %d\n", result);
    } else {
        printf("TD-Lambda training cycle completed successfully\n");
        printf("New model saved to: %s\n", output_model);
    }
}

void test_pytorch(const char *model_path) {
    printf("Testing PyTorch model: %s\n", model_path);

    if (!initialize_pytorch(model_path)) {
        printf("Failed to initialize PyTorch model.\n");
        return;
    }

    // Create a test board
    Board board;
    setup_default_position(&board);

    // Evaluate the position
    float eval = evaluate_pytorch(&board);
    printf("PyTorch evaluation of starting position: %.2f centipawns\n", eval);

    // Make a standard opening move (e4)
    Move e4 = {12, 28, PAWN, EMPTY, EMPTY, 0, 0, 0, 0};
    make_move(&board, &e4);

    // Evaluate again
    eval = evaluate_pytorch(&board);
    printf("PyTorch evaluation after e4: %.2f centipawns\n", eval);

    // Clean up
    shutdown_pytorch();
}

// Update main function to include the new command
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
            cmd_neural_input();
        } else if (strcmp(argv[1], "neural-eval") == 0) {
            // Test neural evaluation
            const char *model_path = (argc > 2) ? argv[2] : "chess_model.onnx";
            cmd_test_neural_model(model_path);
        } else if (strcmp(argv[1], "generate-dataset") == 0) {
            // Generate training dataset
            const char *filename = (argc > 2) ? argv[2] : "chess_dataset.json";
            int num_positions = (argc > 3) ? atoi(argv[3]) : 1000;
            int search_depth = (argc > 4) ? atoi(argv[4]) : 3;
            generate_training_dataset(filename, num_positions, search_depth);
        } else if (strcmp(argv[1], "play-neural") == 0) {
            // Play against computer with neural evaluation
            int depth = (argc > 2) ? atoi(argv[2]) : 3;
            const char *model_path = (argc > 3) ? argv[3] : "chess_model.onnx";
            play_with_neural(model_path, depth);
        } else if (strcmp(argv[1], "td-lambda") == 0) {
            // TD-Lambda training
            const char *initial_model = (argc > 2 && strlen(argv[2]) > 0) ? argv[2] : "chess_model.onnx";
            const char *output_model = (argc > 3) ? argv[3] : "chess_model_improved.onnx";
            int num_games = (argc > 4) ? atoi(argv[4]) : 100;
            float lambda = (argc > 5) ? atof(argv[5]) : 0.7f;
            float temperature = (argc > 6) ? atof(argv[6]) : 1.0f;
            cmd_td_lambda_training(initial_model, output_model, num_games, lambda, temperature);
        } else if (strcmp(argv[1], "generate-self-play") == 0) {
            if (argc < 6) {
                printf("Usage: %s generate-self-play <model_path> <output_path> <num_games> <temperature>\n", argv[0]);
                return 1;
            }

            const char *model_path = argv[2];
            const char *output_path = argv[3];
            int num_games = atoi(argv[4]);
            float temperature = atof(argv[5]);
            unsigned int seed = (argc > 6) ? (unsigned int)atoi(argv[6]) : (unsigned int)time(NULL);

            if (num_games <= 0) {
                printf("Number of games must be positive\n");
                return 1;
            }

            if (temperature <= 0.0f) {
                printf("Temperature must be positive\n");
                return 1;
            }

            bool success = generate_self_play_games(model_path, output_path, num_games, temperature, seed);
            return success ? 0 : 1;
        } else if (strcmp(argv[1], "test-pytorch") == 0) {
            if (argc < 3) {
                printf("Usage: %s test-pytorch <model_path>\n", argv[0]);
                return 1;
            }

            test_pytorch(argv[2]);
            return 0;
        } else if (strcmp(argv[1], "bench") == 0) {
            int hash_size_mb = 16;  // Default hash size in MB
            int depth = 6;          // Default depth

            if (argc > 2) depth = atoi(argv[2]);
            if (argc > 3) hash_size_mb = atoi(argv[3]);

            // Configure search with specified hash size
            uint64_t entries = (hash_size_mb * 1024 * 1024) / sizeof(TTEntry);
            SearchConfig config = DEFAULT_SEARCH_CONFIG;
            config.tt_size = entries;
            config.max_depth = depth;
            config.verbosity = 1;

            if (!init_search(config)) {
                printf("Failed to initialize search with %d MB hash\n", hash_size_mb);
                return 1;
            }

            printf("Running benchmark at depth %d with %d MB hash table...\n", depth, hash_size_mb);

            // Run benchmark on standard test positions
            Board board;
            setup_default_position(&board);

            clock_t start = clock();
            Move best_move;
            uint64_t nodes = 0;

            float score = search_position(&board, depth, &best_move, &nodes);

            clock_t end = clock();
            double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

            char move_str[10];
            strcpy(move_str, move_to_string(best_move));

            printf("Benchmark results:\n");
            printf("Position: Starting position\n");
            printf("Depth: %d\n", depth);
            printf("Hash: %d MB\n", hash_size_mb);
            printf("Best move: %s\n", move_str);
            printf("Score: %.2f centipawns\n", score * 100.0f);
            printf("Nodes: %" PRIu64 "\n", nodes);
            printf("Time: %.3f seconds\n", elapsed);
            printf("NPS: %.0f nodes/second\n", nodes / elapsed);

            cleanup_search();
            return 0;
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
            printf("  play-neural [depth] [model] - Play against computer with neural evaluation\n");
            printf("  td-lambda [initial_model] [output_model] [games] [lambda] - Run TD-Lambda training\n");
            printf("  generate-self-play [model_path] [output_path] [num_games] [temperature] - Generate self-play games\n");
            printf("  bench [depth] [hash_size_mb] - Run benchmark with specified depth and hash size\n");
        }
    } else {
        // Default to interactive mode
        interactive_mode();
    }

    return 0;
}
