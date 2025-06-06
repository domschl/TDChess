#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "board.h"
#include "movegen.h"
#include "perft.h"

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
            make_move(&board, move);
        } else {
            printf("Invalid move! Try again.\n");
        }
    }
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
        } else {
            printf("Unknown command: %s\n", argv[1]);
            printf("Available commands:\n");
            printf("  perft [depth]       - Run perft to specified depth\n");
            printf("  perft-detail [depth] - Show detailed perft results\n");
            printf("  test [max_depth]     - Run perft tests on standard positions\n");
        }
    } else {
        // Default to interactive mode
        interactive_mode();
    }

    return 0;
}
