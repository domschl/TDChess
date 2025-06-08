#include "neural.h"
#include "eval.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#if HAVE_ONNXRUNTIME
// Flag to track ONNX Runtime initialization
static bool onnx_runtime_initialized = false;

// Global singleton evaluator
static NeuralEvaluator* global_evaluator = NULL;

// Get the singleton neural evaluator
NeuralEvaluator* get_neural_evaluator(void) {
    return global_evaluator;
}
#endif

// Convert a board position to a set of planes for neural network input
bool board_to_planes(const Board* board, float* tensor_buffer, size_t buffer_size) {
    if (!board || !tensor_buffer || buffer_size < BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS * sizeof(float)) {
        return false;
    }
    
    // Clear buffer
    memset(tensor_buffer, 0, buffer_size);
    
    // Create tensor representation - 14 planes of 8x8
    // Planes 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
    // Planes 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
    // Plane 12: Side to move (1 for white, 0 for black)
    // Plane 13: En passant square (1 at the square, 0 elsewhere)
    
    // Fill piece planes
    for (int sq = 0; sq < 64; sq++) {
        Piece piece = board->pieces[sq];
        if (piece.type != EMPTY) {
            // Get the plane index
            int plane_idx = (piece.color == WHITE) ? (piece.type - 1) : (piece.type - 1 + 6);
            
            // Set the corresponding position in the tensor
            tensor_buffer[plane_idx * 64 + sq] = 1.0f;
        }
    }
    
    // Fill side to move plane (plane 12)
    if (board->side_to_move == WHITE) {
        for (int sq = 0; sq < 64; sq++) {
            tensor_buffer[12 * 64 + sq] = 1.0f;
        }
    }
    
    // Fill en passant plane (plane 13)
    if (board->en_passant_square >= 0 && board->en_passant_square < 64) {
        tensor_buffer[13 * 64 + board->en_passant_square] = 1.0f;
    }
    
    return true;
}

#if HAVE_ONNXRUNTIME
// Initialize the ONNX runtime and load the model
static bool create_neural_evaluator(const char* model_path) {
    if (global_evaluator) {
        printf("Neural evaluator already initialized\n");
        return true; // Already initialized
    }
    
    // Allocate evaluator
    global_evaluator = (NeuralEvaluator*)malloc(sizeof(NeuralEvaluator));
    if (!global_evaluator) {
        printf("Failed to allocate memory for neural evaluator\n");
        return false;
    }
    
    // Initialize fields
    memset(global_evaluator, 0, sizeof(NeuralEvaluator));
    
    // Get ONNX Runtime API - only do this once
    if (!onnx_runtime_initialized) {
        global_evaluator->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (!global_evaluator->ort) {
            printf("Failed to get ONNX Runtime API\n");
            free(global_evaluator);
            global_evaluator = NULL;
            return false;
        }
        onnx_runtime_initialized = true;
    } else {
        global_evaluator->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    }
    
    // Create environment
    OrtStatus* status = global_evaluator->ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "TDChess", &global_evaluator->env);
    if (status != NULL) {
        const char* error_message = global_evaluator->ort->GetErrorMessage(status);
        printf("Failed to create ONNX Runtime environment: %s\n", error_message);
        global_evaluator->ort->ReleaseStatus(status);
        free(global_evaluator);
        global_evaluator = NULL;
        return false;
    }
    
    // Create session options
    OrtSessionOptions* session_options;
    status = global_evaluator->ort->CreateSessionOptions(&session_options);
    if (status != NULL) {
        const char* error_message = global_evaluator->ort->GetErrorMessage(status);
        printf("Failed to create session options: %s\n", error_message);
        global_evaluator->ort->ReleaseStatus(status);
        global_evaluator->ort->ReleaseEnv(global_evaluator->env);
        free(global_evaluator);
        global_evaluator = NULL;
        return false;
    }
    
    // Fix: Check return value from SetSessionLogVerbosityLevel
    status = global_evaluator->ort->SetSessionLogVerbosityLevel(session_options, ORT_LOGGING_LEVEL_ERROR);
    if (status != NULL) {
        const char* error_message = global_evaluator->ort->GetErrorMessage(status);
        printf("Failed to set session log verbosity level: %s\n", error_message);
        global_evaluator->ort->ReleaseStatus(status);
        global_evaluator->ort->ReleaseSessionOptions(session_options);
        global_evaluator->ort->ReleaseEnv(global_evaluator->env);
        free(global_evaluator);
        global_evaluator = NULL;
        return false;
    }
    
    // Create session
    status = global_evaluator->ort->CreateSession(
        global_evaluator->env,
        model_path,
        session_options,
        &global_evaluator->session
    );
    
    // Release session options regardless of success
    global_evaluator->ort->ReleaseSessionOptions(session_options);
    
    if (status != NULL) {
        const char* error_message = global_evaluator->ort->GetErrorMessage(status);
        printf("Failed to create session: %s\n", error_message);
        global_evaluator->ort->ReleaseStatus(status);
        global_evaluator->ort->ReleaseEnv(global_evaluator->env);
        free(global_evaluator);
        global_evaluator = NULL;
        return false;
    }
    
    // Create memory info
    status = global_evaluator->ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &global_evaluator->memory_info);
    if (status != NULL) {
        const char* error_message = global_evaluator->ort->GetErrorMessage(status);
        printf("Failed to create memory info: %s\n", error_message);
        global_evaluator->ort->ReleaseStatus(status);
        global_evaluator->ort->ReleaseSession(global_evaluator->session);
        global_evaluator->ort->ReleaseEnv(global_evaluator->env);
        free(global_evaluator);
        global_evaluator = NULL;
        return false;
    }
    
    // Set input and output names
    global_evaluator->input_names[0] = "input";
    global_evaluator->output_names[0] = "output";
    
    // Get session information
    size_t num_input_nodes;
    status = global_evaluator->ort->SessionGetInputCount(global_evaluator->session, &num_input_nodes);
    if (status != NULL) {
        global_evaluator->ort->ReleaseStatus(status);
    }
    
    size_t num_output_nodes;
    status = global_evaluator->ort->SessionGetOutputCount(global_evaluator->session, &num_output_nodes);
    if (status != NULL) {
        global_evaluator->ort->ReleaseStatus(status);
    }
    
    printf("Successfully loaded neural model from %s\n", model_path);
    printf("  Inputs: %zu, Outputs: %zu\n", num_input_nodes, num_output_nodes);
    
    return true;
}

// Clean up neural evaluator
static void destroy_neural_evaluator(void) {
    if (!global_evaluator) {
        return; // Already cleaned up
    }
    
    // Release resources in reverse order of creation
    if (global_evaluator->memory_info) {
        global_evaluator->ort->ReleaseMemoryInfo(global_evaluator->memory_info);
    }
    
    if (global_evaluator->session) {
        global_evaluator->ort->ReleaseSession(global_evaluator->session);
    }
    
    if (global_evaluator->env) {
        global_evaluator->ort->ReleaseEnv(global_evaluator->env);
    }
    
    // Free the evaluator structure
    free(global_evaluator);
    global_evaluator = NULL;
}

// Run inference using the neural evaluator
static bool run_neural_inference(float* input_tensor, float* output) {
    if (!global_evaluator || !global_evaluator->session) {
        return false;
    }
    
    // Create input tensor
    OrtValue* input_tensor_ort = NULL;
    size_t input_size = BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS;
    int64_t input_shape[] = {1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE};
    
    OrtStatus* status = global_evaluator->ort->CreateTensorWithDataAsOrtValue(
        global_evaluator->memory_info,
        input_tensor,
        input_size * sizeof(float),
        input_shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor_ort
    );
    
    if (status != NULL) {
        const char* error_message = global_evaluator->ort->GetErrorMessage(status);
        printf("Failed to create input tensor: %s\n", error_message);
        global_evaluator->ort->ReleaseStatus(status);
        return false;
    }
    
    // Create output tensor
    OrtValue* output_tensor_ort = NULL;
    
    // Fix: Use proper const qualifiers for Run function
    const OrtValue* const_input_tensor_ort = input_tensor_ort;
    
    // Run inference
    status = global_evaluator->ort->Run(
        global_evaluator->session,
        NULL, // No run options
        global_evaluator->input_names,
        &const_input_tensor_ort,  // Use const-qualified pointer
        1, // Number of inputs
        global_evaluator->output_names,
        1, // Number of outputs
        &output_tensor_ort
    );
    
    if (status != NULL) {
        const char* error_message = global_evaluator->ort->GetErrorMessage(status);
        printf("Failed to run inference: %s\n", error_message);
        global_evaluator->ort->ReleaseStatus(status);
        global_evaluator->ort->ReleaseValue(input_tensor_ort);
        return false;
    }
    
    // Get output data
    float* output_data;
    status = global_evaluator->ort->GetTensorMutableData(output_tensor_ort, (void**)&output_data);
    if (status != NULL) {
        const char* error_message = global_evaluator->ort->GetErrorMessage(status);
        printf("Failed to get output data: %s\n", error_message);
        global_evaluator->ort->ReleaseStatus(status);
        global_evaluator->ort->ReleaseValue(input_tensor_ort);
        global_evaluator->ort->ReleaseValue(output_tensor_ort);
        return false;
    }
    
    // Copy output value
    *output = output_data[0];
    
    // Clean up
    global_evaluator->ort->ReleaseValue(input_tensor_ort);
    global_evaluator->ort->ReleaseValue(output_tensor_ort);
    
    return true;
}
#endif

// Initialize neural network subsystem
bool initialize_neural(const char* model_path) {
#if HAVE_ONNXRUNTIME
    return create_neural_evaluator(model_path);
#else
    (void)model_path; // Avoid unused parameter warning
    printf("ONNX Runtime support not available\n");
    return false;
#endif
}

// Shutdown neural network subsystem
void shutdown_neural(void) {
#if HAVE_ONNXRUNTIME
    destroy_neural_evaluator();
    onnx_runtime_initialized = false;  // Reset initialization flag
#endif
}

// Evaluate a position using the neural network
// Returns score in CENTIPAWNS from the CURRENT PLAYER's perspective.
float evaluate_neural(const Board* board) {
#if HAVE_ONNXRUNTIME
    if (!global_evaluator || !global_evaluator->session) {
        // Fallback if NN not initialized
        float basic_eval_white_view_pawn_units = evaluate_basic(board); // White's perspective, pawn units
        // Convert to current player's perspective and scale to centipawns
        return (board->side_to_move == WHITE) ? (basic_eval_white_view_pawn_units * 100.0f) : (-basic_eval_white_view_pawn_units * 100.0f);
    }

    // Prepare input tensor
    float input_tensor[BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS];
    if (!board_to_planes(board, input_tensor, sizeof(input_tensor))) {
        // Fallback if board_to_planes fails
        float basic_eval_white_view_pawn_units = evaluate_basic(board);
        return (board->side_to_move == WHITE) ? (basic_eval_white_view_pawn_units * 100.0f) : (-basic_eval_white_view_pawn_units * 100.0f);
    }

    // Run neural evaluation
    float nn_raw_output; // Assume this is in pawn units, from current player's perspective (e.g., tanh output range [-1, 1])
    if (run_neural_inference(input_tensor, &nn_raw_output)) {
        // Scale to centipawns, still from current player's perspective
        return nn_raw_output * 100.0f;
    }
    
    // Fall back to basic evaluation if neural inference fails
    float basic_eval_white_view_pawn_units = evaluate_basic(board);
    return (board->side_to_move == WHITE) ? (basic_eval_white_view_pawn_units * 100.0f) : (-basic_eval_white_view_pawn_units * 100.0f);
#else
    // Fall back to basic evaluation if ONNX Runtime is not available
    float basic_eval_white_view_pawn_units = evaluate_basic(board); // White's perspective, pawn units
    // Convert to current player's perspective and scale to centipawns
    return (board->side_to_move == WHITE) ? (basic_eval_white_view_pawn_units * 100.0f) : (-basic_eval_white_view_pawn_units * 100.0f);
#endif
}

// Test the neural evaluation
void test_neural_evaluation(const Board* board) {
    printf("Neural evaluation: %.3f\n", evaluate_neural(board));
    printf("Classical evaluation: %.3f\n", evaluate_position(board));
}

// Print the neural input representation for debugging
void test_neural_input(void) {
    Board board;
    setup_default_position(&board);
    
    float tensor[BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS];
    board_to_planes(&board, tensor, sizeof(tensor));
    
    printf("Neural input tensor for starting position:\n");
    
    for (int plane = 0; plane < INPUT_CHANNELS; plane++) {
        const char* plane_name;
        switch (plane) {
            case 0: plane_name = "White Pawns"; break;
            case 1: plane_name = "White Knights"; break;
            case 2: plane_name = "White Bishops"; break;
            case 3: plane_name = "White Rooks"; break;
            case 4: plane_name = "White Queens"; break;
            case 5: plane_name = "White King"; break;
            case 6: plane_name = "Black Pawns"; break;
            case 7: plane_name = "Black Knights"; break;
            case 8: plane_name = "Black Bishops"; break;
            case 9: plane_name = "Black Rooks"; break;
            case 10: plane_name = "Black Queens"; break;
            case 11: plane_name = "Black King"; break;
            case 12: plane_name = "Side to Move"; break;
            case 13: plane_name = "En Passant"; break;
            default: plane_name = "Unknown"; break;
        }
        
        printf("Plane %d (%s):\n", plane, plane_name);
        for (int rank = 7; rank >= 0; rank--) {
            for (int file = 0; file < 8; file++) {
                int sq = rank * 8 + file;
                printf("%.0f ", tensor[plane * 64 + sq]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
