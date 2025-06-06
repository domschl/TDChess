#include "neural.h"
#include "eval.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#if HAVE_ONNXRUNTIME
#include <onnxruntime/onnxruntime_c_api.h>
#endif

// Convert a board position to a set of planes for neural network input
void board_to_planes(const Board *board, float *tensor_buffer, size_t buffer_size) {
    // Ensure buffer is large enough
    size_t required_size = BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS * sizeof(float);
    if (buffer_size < required_size) {
        return;  // Buffer too small
    }

    // Clear the buffer
    memset(tensor_buffer, 0, buffer_size);

    // Create planes for each piece type and color
    // Planes are ordered: [wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK, side_to_move, en_passant]

    // Iterate through the board and set the appropriate values
    for (int sq = 0; sq < 64; sq++) {
        Piece piece = board->pieces[sq];
        if (piece.type == EMPTY) continue;

        int rank = SQUARE_RANK(sq);
        int file = SQUARE_FILE(sq);
        int plane_idx = -1;

        // Determine which plane this piece belongs to
        if (piece.color == WHITE) {
            switch (piece.type) {
            case PAWN:
                plane_idx = 0;
                break;
            case KNIGHT:
                plane_idx = 1;
                break;
            case BISHOP:
                plane_idx = 2;
                break;
            case ROOK:
                plane_idx = 3;
                break;
            case QUEEN:
                plane_idx = 4;
                break;
            case KING:
                plane_idx = 5;
                break;
            default:
                break;
            }
        } else {
            switch (piece.type) {
            case PAWN:
                plane_idx = 6;
                break;
            case KNIGHT:
                plane_idx = 7;
                break;
            case BISHOP:
                plane_idx = 8;
                break;
            case ROOK:
                plane_idx = 9;
                break;
            case QUEEN:
                plane_idx = 10;
                break;
            case KING:
                plane_idx = 11;
                break;
            default:
                break;
            }
        }

        // Set the value in the appropriate plane
        if (plane_idx >= 0) {
            int offset = plane_idx * BOARD_SIZE * BOARD_SIZE + rank * BOARD_SIZE + file;
            tensor_buffer[offset] = 1.0f;
        }
    }

    // Set side to move plane (plane 12)
    float side_value = (board->side_to_move == WHITE) ? 1.0f : 0.0f;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        tensor_buffer[12 * BOARD_SIZE * BOARD_SIZE + i] = side_value;
    }

    // Set en passant plane (plane 13)
    if (board->en_passant_square != -1) {
        int ep_rank = SQUARE_RANK(board->en_passant_square);
        int ep_file = SQUARE_FILE(board->en_passant_square);
        int offset = 13 * BOARD_SIZE * BOARD_SIZE + ep_rank * BOARD_SIZE + ep_file;
        tensor_buffer[offset] = 1.0f;
    }
}

// Print the neural input representation for debugging
void print_tensor_representation(const Board *board) {
    // Allocate memory for the tensor
    size_t buffer_size = BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS * sizeof(float);
    float *tensor = (float *)malloc(buffer_size);

    if (!tensor) {
        printf("Memory allocation failed\n");
        return;
    }

    // Convert board to tensor format
    board_to_planes(board, tensor, buffer_size);

    // Print each plane for debugging
    const char *plane_names[] = {
        "White Pawns", "White Knights", "White Bishops", "White Rooks", "White Queens", "White King",
        "Black Pawns", "Black Knights", "Black Bishops", "Black Rooks", "Black Queens", "Black King",
        "Side to move", "En passant"};

    printf("Neural network input tensor representation:\n");
    for (int p = 0; p < INPUT_CHANNELS; p++) {
        printf("\nPlane %d: %s\n", p, plane_names[p]);

        for (int r = 7; r >= 0; r--) {  // Print in reverse order (8->1)
            printf("%d  ", r + 1);      // Rank number
            for (int f = 0; f < 8; f++) {
                int offset = p * BOARD_SIZE * BOARD_SIZE + r * BOARD_SIZE + f;
                printf("%.0f ", tensor[offset]);  // Print 0 or 1
            }
            printf("\n");
        }

        // Print file letters
        printf("   a b c d e f g h\n");
    }

    free(tensor);
}

// Neural evaluator structure
struct NeuralEvaluator {
    char *model_path;

#if HAVE_ONNXRUNTIME
    // ONNX Runtime components
    const OrtApi *ort;
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *memory_info;
    OrtAllocator *allocator;  // Default allocator (don't free this)

    // Input/output information
    size_t input_count;
    size_t output_count;
    char **input_names;
    char **output_names;
#endif
};

// Global neural evaluator instance
static NeuralEvaluator *global_evaluator = NULL;

// Check if neural network support is available
bool is_neural_available(void) {
#if HAVE_ONNXRUNTIME
    return true;
#else
    return false;
#endif
}

// Load a neural evaluator from an ONNX model file
NeuralEvaluator *load_neural_evaluator(const char *model_path) {
#if HAVE_ONNXRUNTIME
    NeuralEvaluator *evaluator = (NeuralEvaluator *)malloc(sizeof(NeuralEvaluator));
    if (!evaluator) {
        printf("Failed to allocate memory for neural evaluator\n");
        return NULL;
    }

    // Initialize with zeros
    memset(evaluator, 0, sizeof(NeuralEvaluator));

    // Copy model path
    evaluator->model_path = strdup(model_path);
    if (!evaluator->model_path) {
        printf("Failed to allocate memory for model path\n");
        free(evaluator);
        return NULL;
    }

    // Get OrtApi
    evaluator->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!evaluator->ort) {
        printf("Failed to get ONNX Runtime API\n");
        free(evaluator->model_path);
        free(evaluator);
        return NULL;
    }

    // Create environment
    OrtStatus *status = evaluator->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "TDChess", &evaluator->env);
    if (status != NULL) {
        const char *error_message = evaluator->ort->GetErrorMessage(status);
        printf("Failed to create ONNX Runtime environment: %s\n", error_message);
        evaluator->ort->ReleaseStatus(status);
        free(evaluator->model_path);
        free(evaluator);
        return NULL;
    }

    // Get default allocator
    status = evaluator->ort->GetAllocatorWithDefaultOptions(&evaluator->allocator);
    if (status != NULL) {
        const char *error_message = evaluator->ort->GetErrorMessage(status);
        printf("Failed to get default allocator: %s\n", error_message);
        evaluator->ort->ReleaseStatus(status);
        evaluator->ort->ReleaseEnv(evaluator->env);
        free(evaluator->model_path);
        free(evaluator);
        return NULL;
    }

    // Create session options
    OrtSessionOptions *session_options;
    status = evaluator->ort->CreateSessionOptions(&session_options);
    if (status != NULL) {
        const char *error_message = evaluator->ort->GetErrorMessage(status);
        printf("Failed to create session options: %s\n", error_message);
        evaluator->ort->ReleaseStatus(status);
        evaluator->ort->ReleaseEnv(evaluator->env);
        free(evaluator->model_path);
        free(evaluator);
        return NULL;
    }

    // Set graph optimization level
    status = evaluator->ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);
    if (status != NULL) {
        const char *error_message = evaluator->ort->GetErrorMessage(status);
        printf("Warning: Failed to set graph optimization level: %s\n", error_message);
        evaluator->ort->ReleaseStatus(status);
        // Continue anyway as this is not critical
    }

    // Create session
    status = evaluator->ort->CreateSession(evaluator->env, model_path, session_options, &evaluator->session);
    evaluator->ort->ReleaseSessionOptions(session_options);

    if (status != NULL) {
        const char *error_message = evaluator->ort->GetErrorMessage(status);
        printf("Failed to create session: %s\n", error_message);
        evaluator->ort->ReleaseStatus(status);
        evaluator->ort->ReleaseEnv(evaluator->env);
        free(evaluator->model_path);
        free(evaluator);
        return NULL;
    }

    // Create memory info
    status = evaluator->ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &evaluator->memory_info);
    if (status != NULL) {
        const char *error_message = evaluator->ort->GetErrorMessage(status);
        printf("Failed to create memory info: %s\n", error_message);
        evaluator->ort->ReleaseStatus(status);
        evaluator->ort->ReleaseSession(evaluator->session);
        evaluator->ort->ReleaseEnv(evaluator->env);
        free(evaluator->model_path);
        free(evaluator);
        return NULL;
    }

    // Get input and output count
    status = evaluator->ort->SessionGetInputCount(evaluator->session, &evaluator->input_count);
    if (status != NULL) {
        const char *error_message = evaluator->ort->GetErrorMessage(status);
        printf("Failed to get input count: %s\n", error_message);
        evaluator->ort->ReleaseStatus(status);
        evaluator->ort->ReleaseMemoryInfo(evaluator->memory_info);
        evaluator->ort->ReleaseSession(evaluator->session);
        evaluator->ort->ReleaseEnv(evaluator->env);
        free(evaluator->model_path);
        free(evaluator);
        return NULL;
    }

    status = evaluator->ort->SessionGetOutputCount(evaluator->session, &evaluator->output_count);
    if (status != NULL) {
        const char *error_message = evaluator->ort->GetErrorMessage(status);
        printf("Failed to get output count: %s\n", error_message);
        evaluator->ort->ReleaseStatus(status);
        evaluator->ort->ReleaseMemoryInfo(evaluator->memory_info);
        evaluator->ort->ReleaseSession(evaluator->session);
        evaluator->ort->ReleaseEnv(evaluator->env);
        free(evaluator->model_path);
        free(evaluator);
        return NULL;
    }

    // Allocate memory for input and output names
    evaluator->input_names = (char **)malloc(sizeof(char *) * evaluator->input_count);
    evaluator->output_names = (char **)malloc(sizeof(char *) * evaluator->output_count);

    if (!evaluator->input_names || !evaluator->output_names) {
        printf("Failed to allocate memory for input/output names\n");
        if (evaluator->input_names) free(evaluator->input_names);
        if (evaluator->output_names) free(evaluator->output_names);
        evaluator->ort->ReleaseMemoryInfo(evaluator->memory_info);
        evaluator->ort->ReleaseSession(evaluator->session);
        evaluator->ort->ReleaseEnv(evaluator->env);
        free(evaluator->model_path);
        free(evaluator);
        return NULL;
    }

    // Get input and output names
    for (size_t i = 0; i < evaluator->input_count; i++) {
        char *input_name;
        status = evaluator->ort->SessionGetInputName(evaluator->session, i, evaluator->allocator, &input_name);
        if (status != NULL) {
            const char *error_message = evaluator->ort->GetErrorMessage(status);
            printf("Failed to get input name: %s\n", error_message);
            evaluator->ort->ReleaseStatus(status);
            // Clean up
            for (size_t j = 0; j < i; j++) {
                OrtStatus *free_status = evaluator->ort->AllocatorFree(evaluator->allocator, evaluator->input_names[j]);
                if (free_status != NULL) {
                    evaluator->ort->ReleaseStatus(free_status);
                }
            }
            free(evaluator->input_names);
            free(evaluator->output_names);
            evaluator->ort->ReleaseMemoryInfo(evaluator->memory_info);
            evaluator->ort->ReleaseSession(evaluator->session);
            evaluator->ort->ReleaseEnv(evaluator->env);
            free(evaluator->model_path);
            free(evaluator);
            return NULL;
        }
        evaluator->input_names[i] = input_name;
    }

    for (size_t i = 0; i < evaluator->output_count; i++) {
        char *output_name;
        status = evaluator->ort->SessionGetOutputName(evaluator->session, i, evaluator->allocator, &output_name);
        if (status != NULL) {
            const char *error_message = evaluator->ort->GetErrorMessage(status);
            printf("Failed to get output name: %s\n", error_message);
            evaluator->ort->ReleaseStatus(status);
            // Clean up
            for (size_t j = 0; j < evaluator->input_count; j++) {
                OrtStatus *free_status = evaluator->ort->AllocatorFree(evaluator->allocator, evaluator->input_names[j]);
                if (free_status != NULL) {
                    evaluator->ort->ReleaseStatus(free_status);
                }
            }
            for (size_t j = 0; j < i; j++) {
                OrtStatus *free_status = evaluator->ort->AllocatorFree(evaluator->allocator, evaluator->output_names[j]);
                if (free_status != NULL) {
                    evaluator->ort->ReleaseStatus(free_status);
                }
            }
            free(evaluator->input_names);
            free(evaluator->output_names);
            evaluator->ort->ReleaseMemoryInfo(evaluator->memory_info);
            evaluator->ort->ReleaseSession(evaluator->session);
            evaluator->ort->ReleaseEnv(evaluator->env);
            free(evaluator->model_path);
            free(evaluator);
            return NULL;
        }
        evaluator->output_names[i] = output_name;
    }

    printf("Successfully loaded neural model from %s\n", model_path);
    printf("  Inputs: %zu, Outputs: %zu\n", evaluator->input_count, evaluator->output_count);

    return evaluator;
#else
    printf("Neural network support is not available (ONNX Runtime not found)\n");
    return NULL;
#endif
}

// Free a neural evaluator
void free_neural_evaluator(NeuralEvaluator *evaluator) {
    if (!evaluator) return;

#if HAVE_ONNXRUNTIME
    if (evaluator->input_names) {
        for (size_t i = 0; i < evaluator->input_count; i++) {
            OrtStatus *status = evaluator->ort->AllocatorFree(evaluator->allocator, evaluator->input_names[i]);
            if (status != NULL) {
                evaluator->ort->ReleaseStatus(status);
            }
        }
        free(evaluator->input_names);
    }

    if (evaluator->output_names) {
        for (size_t i = 0; i < evaluator->output_count; i++) {
            OrtStatus *status = evaluator->ort->AllocatorFree(evaluator->allocator, evaluator->output_names[i]);
            if (status != NULL) {
                evaluator->ort->ReleaseStatus(status);
            }
        }
        free(evaluator->output_names);
    }

    if (evaluator->memory_info) {
        evaluator->ort->ReleaseMemoryInfo(evaluator->memory_info);
    }

    if (evaluator->session) {
        evaluator->ort->ReleaseSession(evaluator->session);
    }

    if (evaluator->env) {
        evaluator->ort->ReleaseEnv(evaluator->env);
    }

    // Note: We don't free allocator as it's a default allocator provided by ONNX Runtime
    // According to the documentation: "Returned value should NOT be freed"
#endif

    free(evaluator->model_path);
    free(evaluator);
}

// Evaluate a position using the neural evaluator
float neural_evaluate_position(NeuralEvaluator *evaluator, const Board *board) {
#if HAVE_ONNXRUNTIME
    if (!evaluator || !evaluator->session) {
        return evaluate_position(board);
    }

    // Prepare input tensor
    size_t tensor_size = BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS;
    float *input_tensor = (float *)malloc(tensor_size * sizeof(float));
    if (!input_tensor) {
        printf("Failed to allocate memory for input tensor\n");
        return evaluate_position(board);
    }

    // Convert board to tensor
    board_to_planes(board, input_tensor, tensor_size * sizeof(float));

    // Create input tensor
    int64_t input_shape[4] = {1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE};
    OrtValue *input_tensor_ort = NULL;
    OrtStatus *status = evaluator->ort->CreateTensorWithDataAsOrtValue(
        evaluator->memory_info, input_tensor, tensor_size * sizeof(float),
        input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor_ort);

    if (status != NULL) {
        const char *error_message = evaluator->ort->GetErrorMessage(status);
        printf("Failed to create input tensor: %s\n", error_message);
        evaluator->ort->ReleaseStatus(status);
        free(input_tensor);
        return evaluate_position(board);
    }

    // Create output tensor
    OrtValue *output_tensor = NULL;

    // Prepare input values array for Run
    const OrtValue *input_values[1] = {input_tensor_ort};

    // Run inference
    status = evaluator->ort->Run(
        evaluator->session, NULL,
        (const char *const *)evaluator->input_names, input_values, 1,
        (const char *const *)evaluator->output_names, 1, &output_tensor);

    // Clean up input tensor
    evaluator->ort->ReleaseValue(input_tensor_ort);
    free(input_tensor);

    if (status != NULL) {
        const char *error_message = evaluator->ort->GetErrorMessage(status);
        printf("Failed to run inference: %s\n", error_message);
        evaluator->ort->ReleaseStatus(status);
        return evaluate_position(board);
    }

    // Get output data
    float *output_data;
    status = evaluator->ort->GetTensorMutableData(output_tensor, (void **)&output_data);
    if (status != NULL) {
        const char *error_message = evaluator->ort->GetErrorMessage(status);
        printf("Failed to get output data: %s\n", error_message);
        evaluator->ort->ReleaseStatus(status);
        evaluator->ort->ReleaseValue(output_tensor);
        return evaluate_position(board);
    }

    // Get the evaluation score
    float score = output_data[0];

    // Clean up
    evaluator->ort->ReleaseValue(output_tensor);

    return score;
#else
    // Neural evaluation not available, fall back to basic evaluation
    return evaluate_position(board);
#endif
}

// Set the global neural evaluator
void set_neural_evaluator(NeuralEvaluator *evaluator) {
    if (global_evaluator) {
        free_neural_evaluator(global_evaluator);
    }
    global_evaluator = evaluator;
}

// Get the global neural evaluator
NeuralEvaluator *get_neural_evaluator(void) {
    return global_evaluator;
}
