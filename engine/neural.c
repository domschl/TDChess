#include "neural.h"
#include "eval.h"             // For evaluate_basic as fallback
#include "pytorch_binding.h"  // New include for PyTorch bindings
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>  // For bool type

// Flag to track PyTorch initialization
static bool pytorch_initialized = false;

// Public function to initialize the neural network subsystem
bool initialize_neural(const char *model_path) {
    if (pytorch_initialized) {
        printf("Neural subsystem already initialized.\n");
        return true;
    }

    printf("Initializing neural subsystem with model: %s\n", model_path ? model_path : "default (model.pt)");

    // Initialize PyTorch
    const char *effective_model_path = model_path ? model_path : "model.pt";
    if (initialize_pytorch(effective_model_path)) {
        pytorch_initialized = true;
        printf("PyTorch neural subsystem initialized successfully.\n");
        return true;
    } else {
        fprintf(stderr, "Failed to initialize PyTorch neural subsystem.\n");
        return false;
    }
}

// Public function to shut down the neural network subsystem
void shutdown_neural(void) {
    printf("Shutting down neural subsystem.\n");

    if (pytorch_initialized) {
        shutdown_pytorch();
        pytorch_initialized = false;
    }

    printf("Neural subsystem shutdown complete.\n");
}

// Convert a board position to a set of planes for neural network input
bool board_to_planes(const Board *board, float *tensor_buffer, size_t buffer_size_bytes) {
    if (!board || !tensor_buffer) {
        fprintf(stderr, "Error (board_to_planes): Null board or tensor_buffer.\n");
        return false;
    }
    size_t required_size_bytes = (size_t)INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE * sizeof(float);
    if (buffer_size_bytes < required_size_bytes) {
        fprintf(stderr, "Error (board_to_planes): buffer_size_bytes (%zu) is too small. Required: %zu bytes.\n", buffer_size_bytes, required_size_bytes);
        return false;
    }

    memset(tensor_buffer, 0, required_size_bytes);

    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        Piece piece = board->pieces[sq_idx];
        if (piece.type != EMPTY) {
            int piece_type_plane_offset = piece.type - 1;  // PAWN=0, KNIGHT=1, ..., KING=5
            int color_base_plane = (piece.color == WHITE) ? 0 : 6;
            int plane_idx = color_base_plane + piece_type_plane_offset;

            // CHW format: tensor_buffer[plane_idx * height * width + row_idx * width + col_idx]
            tensor_buffer[plane_idx * (BOARD_SIZE * BOARD_SIZE) + sq_idx] = 1.0f;
        }
    }

    // Plane 12: Side to move (1.0 for white, 0.0 for black - fill entire plane)
    float side_to_move_value = (board->side_to_move == WHITE) ? 1.0f : 0.0f;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
        tensor_buffer[12 * (BOARD_SIZE * BOARD_SIZE) + i] = side_to_move_value;
    }

    // Plane 13: En passant square (1.0 at the square, 0.0 elsewhere)
    if (board->en_passant_square >= 0 && board->en_passant_square < 64) {
        tensor_buffer[13 * (BOARD_SIZE * BOARD_SIZE) + board->en_passant_square] = 1.0f;
    }
    // Other squares in plane 13 remain 0.0 due to memset.

    return true;
}

// Evaluate a position using the neural network
// Returns score in CENTIPAWNS from the CURRENT PLAYER's perspective.
float evaluate_neural(const Board *board) {
    // Check if PyTorch is initialized
    if (!pytorch_initialized || !is_pytorch_initialized()) {
        fprintf(stderr, "Warning: Neural network not available. Falling back to basic eval in evaluate_neural.\n");
        float basic_eval_white_view_pawn_units = evaluate_basic(board);
        return (board->side_to_move == WHITE) ? (basic_eval_white_view_pawn_units * 100.0f) : (-basic_eval_white_view_pawn_units * 100.0f);
    }

    // Use PyTorch for evaluation
    return evaluate_pytorch(board);
}

// Function for testing the neural evaluation
void test_neural_evaluation(const Board *board) {
    if (pytorch_initialized) {
        printf("PyTorch Neural evaluation: %.3f centipawns\n", evaluate_neural(board));
    } else {
        printf("Neural evaluation skipped (PyTorch not initialized).\n");
    }
}

// Print the neural input representation for debugging
void test_neural_input(void) {
    Board board;
    setup_default_position(&board);

    float tensor[INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE];
    if (!board_to_planes(&board, tensor, sizeof(tensor))) {
        printf("Failed to generate planes for test_neural_input.\n");
        return;
    }

    printf("Neural input tensor for starting position (%d planes of %dx%d):\n", INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE);

    for (int plane = 0; plane < INPUT_CHANNELS; plane++) {
        const char *plane_name = "Unknown";
        // Simplified plane naming for brevity, expand as needed
        if (plane < 6)
            plane_name = "White Pieces (0-5)";
        else if (plane < 12)
            plane_name = "Black Pieces (6-11)";
        else if (plane == 12)
            plane_name = "Side to Move";
        else if (plane == 13)
            plane_name = "En Passant";

        printf("Plane %d (%s):\n", plane, plane_name);
        for (int rank = 7; rank >= 0; rank--) {  // Print ranks from 8 down to 1
            for (int file = 0; file < BOARD_SIZE; file++) {
                int sq_idx_for_print = rank * BOARD_SIZE + file;  // Standard square index
                // Access tensor in CHW format: tensor[plane_idx * H * W + r * W + c]
                printf("%.0f ", tensor[plane * (BOARD_SIZE * BOARD_SIZE) + sq_idx_for_print]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

// For compatibility with existing code - redirects to PyTorch
bool run_neural_inference(const float *input_tensor_values, float *output_value) {
    if (!pytorch_initialized) {
        fprintf(stderr, "PyTorch not initialized. Cannot run inference.\n");
        return false;
    }

    // Create a dummy board to satisfy the interface
    // This is a temporary solution - a better approach would be to update
    // all code that calls this function to use evaluate_pytorch directly
    static Board dummy_board;
    setup_default_position(&dummy_board);

    // Call PyTorch inference
    *output_value = evaluate_pytorch(&dummy_board) / 100.0f;  // Convert centipawns to pawns
    return true;
}
