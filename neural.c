#include "neural.h"
#include "eval.h"  // For evaluate_basic as fallback
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>  // For bool type

// HAVE_ONNXRUNTIME is checked by the include guard in neural.h for OrtApi, etc.
// and also used to conditionally compile functions here.

#if HAVE_ONNXRUNTIME

// Global ONNX Runtime API structure
static const OrtApi *g_ort_api = NULL;

// Flag to track ONNX Runtime initialization
static bool onnx_runtime_fully_initialized = false;       // Tracks if global_evaluator is fully set up
static OrtEnv *ort_env_global = NULL;                     // Renamed to avoid conflict if NeuralEvaluator also has 'env'
static OrtSessionOptions *session_options_global = NULL;  // Renamed
static OrtMemoryInfo *memory_info_global = NULL;          // Global memory info for CPU

// Global singleton evaluator
static NeuralEvaluator *global_evaluator_singleton = NULL;  // Renamed for clarity

// Forward declaration for the internal implementation
static bool initialize_neural_evaluator_internal(const char *model_path);
static bool run_neural_inference_impl(const float *input_tensor_values, float *output_value);

// Helper to ensure g_ort_api is initialized
static bool ensure_ort_api_is_initialized(void) {
    if (!g_ort_api) {
        g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (!g_ort_api) {
            fprintf(stderr, "Error: Failed to get ONNX Runtime API base. ORT_API_VERSION: %d\n", ORT_API_VERSION);
            return false;
        }
    }
    return true;
}

// Public function to initialize the neural network subsystem
bool initialize_neural(const char *model_path) {
    if (onnx_runtime_fully_initialized && global_evaluator_singleton && global_evaluator_singleton->session) {
        printf("Neural subsystem already initialized.\n");
        return true;
    }
    printf("Initializing neural subsystem with model: %s\n", model_path ? model_path : "default (model.onnx)");
    return initialize_neural_evaluator_internal(model_path);
}

// Public function to shut down the neural network subsystem
void shutdown_neural(void) {
    printf("Shutting down neural subsystem.\n");
    if (!ensure_ort_api_is_initialized()) {
        fprintf(stderr, "Warning: Cannot shutdown neural subsystem, ONNX Runtime API not available (g_ort_api is NULL).\n");
        if (global_evaluator_singleton) {
            free(global_evaluator_singleton);
            global_evaluator_singleton = NULL;
        }
        session_options_global = NULL;
        ort_env_global = NULL;
        if (memory_info_global) { /* No explicit g_ort_api->ReleaseMemoryInfo for global one unless created per session */
        }
        memory_info_global = NULL;
        onnx_runtime_fully_initialized = false;
        return;
    }

    if (global_evaluator_singleton) {
        if (global_evaluator_singleton->session) {
            g_ort_api->ReleaseSession(global_evaluator_singleton->session);
            global_evaluator_singleton->session = NULL;
        }
        // memory_info is part of global_evaluator_singleton struct, but owned globally here
        if (global_evaluator_singleton->memory_info && global_evaluator_singleton->memory_info == memory_info_global) {
            // Do not release it here if it's the global one, release memory_info_global below
        }
        free(global_evaluator_singleton);
        global_evaluator_singleton = NULL;
    }
    if (session_options_global) {
        g_ort_api->ReleaseSessionOptions(session_options_global);
        session_options_global = NULL;
    }
    if (memory_info_global) {  // Release the globally created memory_info
        g_ort_api->ReleaseMemoryInfo(memory_info_global);
        memory_info_global = NULL;
    }
    if (ort_env_global) {
        g_ort_api->ReleaseEnv(ort_env_global);
        ort_env_global = NULL;
    }

    onnx_runtime_fully_initialized = false;
    printf("Neural subsystem shutdown complete.\n");
}

// Initialize the neural evaluator (singleton) - internal implementation
static bool initialize_neural_evaluator_internal(const char *model_path) {
    if (onnx_runtime_fully_initialized && global_evaluator_singleton && global_evaluator_singleton->session) {
        return true;
    }

    if (!ensure_ort_api_is_initialized()) {
        return false;
    }

    // Clean up any previous (potentially partial) initialization
    if (global_evaluator_singleton) free(global_evaluator_singleton);
    global_evaluator_singleton = NULL;
    if (session_options_global) g_ort_api->ReleaseSessionOptions(session_options_global);
    session_options_global = NULL;
    if (memory_info_global) g_ort_api->ReleaseMemoryInfo(memory_info_global);
    memory_info_global = NULL;
    if (ort_env_global) g_ort_api->ReleaseEnv(ort_env_global);
    ort_env_global = NULL;
    onnx_runtime_fully_initialized = false;

    OrtStatus *status = NULL;  // ORT_OK is NULL for status pointers

    status = g_ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "TDChessORT", &ort_env_global);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to create ONNX Runtime environment: %s\n", g_ort_api->GetErrorMessage(status));
        g_ort_api->ReleaseStatus(status);
        return false;
    }

    status = g_ort_api->CreateSessionOptions(&session_options_global);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to create ONNX Runtime session options: %s\n", g_ort_api->GetErrorMessage(status));
        g_ort_api->ReleaseStatus(status);
        g_ort_api->ReleaseEnv(ort_env_global);
        ort_env_global = NULL;
        return false;
    }

    bool provider_set = false;
#ifdef __APPLE__
    printf("Info: Attempting to use CoreML Execution Provider (macOS).\n");
    uint32_t coreml_flags = 0;
#ifdef ORT_COREML_FLAG_ENABLE_ON_SUBGRAPH
    coreml_flags = ORT_COREML_FLAG_ENABLE_ON_SUBGRAPH;
    printf("Info: Using ORT_COREML_FLAG_ENABLE_ON_SUBGRAPH (%u).\n", coreml_flags);
#else
    coreml_flags = 2;
    printf("Warning: ORT_COREML_FLAG_ENABLE_ON_SUBGRAPH macro not defined by headers, using fallback value %u for CoreML flags.\n", coreml_flags);
#endif

    // CRITICAL: The error "no member named 'SessionOptionsAppendExecutionProvider_CoreML' in 'struct OrtApi'"
    // means this specific function is NOT on the g_ort_api dispatch table for your ONNX Runtime build.
    // You MUST find the correct way to enable CoreML for your ORT version.
    // This might involve a different function name, or it might be a standalone function not on g_ort_api.
    // For example, it could be a function like `OrtApis::SessionOptionsAppendExecutionProvider_CoreML` (hypothetical).
    // Check your specific onnxruntime_c_api.h or related EP headers.
    // If this call is incorrect, it will not compile or will fail at runtime.
    // status = g_ort_api->SessionOptionsAppendExecutionProvider_CoreML(session_options_global, coreml_flags);
    // For now, I will comment this out to prevent compilation failure. You need to resolve this.
    fprintf(stderr, "Warning: CoreML EP enabling is commented out due to potential API incompatibility. Please verify for your ONNX Runtime version.\n");
    status = NULL;  // Assume it's not set or failed for now.

    if (status == NULL) {  // ORT_OK
        // printf("Info: CoreML Execution Provider appended successfully.\n"); // If the call was made and successful
        // provider_set = true;
    } else {
        // fprintf(stderr, "Warning: Failed to append CoreML Execution Provider: %s. Will use CPU.\n", g_ort_api->GetErrorMessage(status));
        // g_ort_api->ReleaseStatus(status);
    }
#elif __linux__
    printf("Info: Attempting to use CUDA Execution Provider (Linux).\n");
    OrtCUDAProviderOptionsV2 *cuda_options = NULL;
    status = g_ort_api->SessionOptionsAppendExecutionProvider_CUDA_V2(session_options_global, cuda_options);
    if (status == NULL) {
        printf("Info: CUDA Execution Provider appended successfully.\n");
        provider_set = true;
    } else {
        fprintf(stderr, "Warning: Failed to append CUDA Execution Provider: %s. Will use CPU.\n", g_ort_api->GetErrorMessage(status));
        g_ort_api->ReleaseStatus(status);
    }
#endif

    if (!provider_set) {
        printf("Info: Using default CPU Execution Provider.\n");
    }

    status = g_ort_api->SetSessionGraphOptimizationLevel(session_options_global, ORT_ENABLE_ALL);
    if (status != NULL) {
        fprintf(stderr, "Warning: Failed to set graph optimization level: %s\n", g_ort_api->GetErrorMessage(status));
        g_ort_api->ReleaseStatus(status);
    }

    status = g_ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_global);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to create CPU memory info: %s\n", g_ort_api->GetErrorMessage(status));
        g_ort_api->ReleaseStatus(status);
        g_ort_api->ReleaseSessionOptions(session_options_global);
        session_options_global = NULL;
        g_ort_api->ReleaseEnv(ort_env_global);
        ort_env_global = NULL;
        return false;
    }

    global_evaluator_singleton = (NeuralEvaluator *)malloc(sizeof(NeuralEvaluator));
    if (!global_evaluator_singleton) {
        fprintf(stderr, "Error: Failed to allocate memory for NeuralEvaluator.\n");
        g_ort_api->ReleaseMemoryInfo(memory_info_global);
        memory_info_global = NULL;
        g_ort_api->ReleaseSessionOptions(session_options_global);
        session_options_global = NULL;
        g_ort_api->ReleaseEnv(ort_env_global);
        ort_env_global = NULL;
        return false;
    }

    // Populate NeuralEvaluator struct from neural.h
    global_evaluator_singleton->ort = g_ort_api;
    global_evaluator_singleton->env = ort_env_global;
    global_evaluator_singleton->session = NULL;  // Session created next
    global_evaluator_singleton->memory_info = memory_info_global;
    global_evaluator_singleton->input_names[0] = "input";    // CORRECTED: Match ONNX export
    global_evaluator_singleton->output_names[0] = "output";  // CORRECTED: Match ONNX export

    const char *effective_model_path = model_path ? model_path : "model.onnx";
    printf("Info: Loading ONNX model from: %s\n", effective_model_path);
    status = g_ort_api->CreateSession(ort_env_global, effective_model_path, session_options_global, &global_evaluator_singleton->session);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to create ONNX Runtime session for model '%s': %s\n", effective_model_path, g_ort_api->GetErrorMessage(status));
        g_ort_api->ReleaseStatus(status);
        free(global_evaluator_singleton);
        global_evaluator_singleton = NULL;
        g_ort_api->ReleaseMemoryInfo(memory_info_global);
        memory_info_global = NULL;
        g_ort_api->ReleaseSessionOptions(session_options_global);
        session_options_global = NULL;
        g_ort_api->ReleaseEnv(ort_env_global);
        ort_env_global = NULL;
        return false;
    }

    printf("Info: ONNX Runtime session created successfully with model: %s.\n", effective_model_path);
    onnx_runtime_fully_initialized = true;
    return true;
}

// Get the singleton neural evaluator - public function from neural.h
NeuralEvaluator *get_neural_evaluator(void) {
    if (!onnx_runtime_fully_initialized) {
        // This function is now just a getter. Initialization is done by initialize_neural.
        // If called before initialize_neural, it should ideally return NULL or an uninitialized state.
        // However, to maintain some previous behavior of auto-init, we can call initialize_neural.
        // This might not be ideal if model_path needs to be specific.
        printf("Warning: get_neural_evaluator() called before explicit initialize_neural(). Attempting default init.\n");
        if (!initialize_neural("model.onnx")) {  // Or pass NULL for default path handling in initialize_neural_evaluator_internal
            fprintf(stderr, "Error: Default initialization failed in get_neural_evaluator.\n");
            return NULL;
        }
    }
    return global_evaluator_singleton;
}

// Internal implementation for neural inference
static bool run_neural_inference_impl(const float *input_tensor_values, float *output_value) {
    if (!onnx_runtime_fully_initialized || !global_evaluator_singleton || !global_evaluator_singleton->session || !global_evaluator_singleton->ort || !global_evaluator_singleton->memory_info) {
        fprintf(stderr, "Error: Neural evaluator not ready for inference (components missing).\n");
        return false;
    }

    const OrtApi *ort = global_evaluator_singleton->ort;  // Use from struct
    OrtStatus *status = NULL;
    OrtValue *input_tensor_ort = NULL;
    OrtValue *output_tensor_ort = NULL;

    const int64_t input_shape[] = {1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE};
    size_t input_tensor_size_elements = 1 * INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;

    status = ort->CreateTensorWithDataAsOrtValue(global_evaluator_singleton->memory_info, (void *)input_tensor_values,
                                                 input_tensor_size_elements * sizeof(float),
                                                 input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                 &input_tensor_ort);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to create input tensor: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        return false;
    }

    const OrtValue *const_input_tensor_ort_ptr = input_tensor_ort;
    status = ort->Run(global_evaluator_singleton->session, NULL,
                      global_evaluator_singleton->input_names, &const_input_tensor_ort_ptr, 1,
                      global_evaluator_singleton->output_names, 1,
                      &output_tensor_ort);

    if (status != NULL) {
        fprintf(stderr, "Error: Failed to run ONNX inference: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        ort->ReleaseValue(input_tensor_ort);
        return false;
    }

    float *output_tensor_values_ptr;
    status = ort->GetTensorMutableData(output_tensor_ort, (void **)&output_tensor_values_ptr);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get output tensor data: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
        ort->ReleaseValue(output_tensor_ort);
        ort->ReleaseValue(input_tensor_ort);
        return false;
    }
    *output_value = output_tensor_values_ptr[0];  // Assuming scalar output

    ort->ReleaseValue(output_tensor_ort);
    ort->ReleaseValue(input_tensor_ort);
    return true;
}

#else  // HAVE_ONNXRUNTIME is false

// Stubs for when ONNX Runtime is not available
bool initialize_neural(const char *model_path) {
    (void)model_path;
    printf("ONNX Runtime support not available. Neural features disabled.\n");
    return false;
}

void shutdown_neural(void) {
    printf("ONNX Runtime support not available. No neural resources to shut down.\n");
}

NeuralEvaluator *get_neural_evaluator(void) {
    printf("ONNX Runtime support not available. Cannot get neural evaluator.\n");
    return NULL;
}

static bool run_neural_inference_impl(const float *input_tensor_values, float *output_value) {
    (void)input_tensor_values;
    (void)output_value;
    fprintf(stderr, "ONNX Runtime support not available. Cannot run inference.\n");
    return false;
}

#endif  // HAVE_ONNXRUNTIME

// Convert a board position to a set of planes for neural network input
// This function is independent of ONNX runtime itself, so it's outside the #if HAVE_ONNXRUNTIME block for its body
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
            // sq_idx is already effectively row_major (rank * 8 + file) if BOARD_SIZE is 8
            // For CHW, if sq_idx is (rank * 8 + file), then the index is:
            // plane_idx * (BOARD_SIZE * BOARD_SIZE) + rank * BOARD_SIZE + file
            // Since sq_idx = rank * BOARD_SIZE + file (assuming BOARD_SIZE=8)
            tensor_buffer[plane_idx * (BOARD_SIZE * BOARD_SIZE) + sq_idx] = 1.0f;
        }
    }

    // Plane 12: Side to move (1.0 for white, 0.0 for black - fill entire plane)
    float side_to_move_value = (board->side_to_move == WHITE) ? 1.0f : 0.0f;  // Changed to 0.0 for black
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
#if HAVE_ONNXRUNTIME
    NeuralEvaluator *evaluator = get_neural_evaluator();
    if (!evaluator || !evaluator->session) {  // Check if initialization was successful
        fprintf(stderr, "Warning: Neural network not available. Falling back to basic eval in evaluate_neural.\n");
        float basic_eval_white_view_pawn_units = evaluate_basic(board);
        return (board->side_to_move == WHITE) ? (basic_eval_white_view_pawn_units * 100.0f) : (-basic_eval_white_view_pawn_units * 100.0f);
    }

    float input_tensor_values[INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE];
    if (!board_to_planes(board, input_tensor_values, sizeof(input_tensor_values))) {
        fprintf(stderr, "Warning: Failed to convert board to planes. Falling back to basic eval.\n");
        float basic_eval_white_view_pawn_units = evaluate_basic(board);
        return (board->side_to_move == WHITE) ? (basic_eval_white_view_pawn_units * 100.0f) : (-basic_eval_white_view_pawn_units * 100.0f);
    }

    float nn_raw_output;
    if (run_neural_inference_impl(input_tensor_values, &nn_raw_output)) {
        // Assuming nn_raw_output is already from current player's perspective.
        // If your NN outputs, for example, a value in [-1, 1] (from tanh)
        // representing win probability or scaled eval, you might scale it here.
        // Example: return nn_raw_output * 500.0f; // Scale to a centipawn range
        return nn_raw_output * 100.0f;  // Assuming output is in pawn units, convert to centipawns
    }

    fprintf(stderr, "Warning: Neural inference failed. Falling back to basic eval.\n");
    float basic_eval_white_view_pawn_units = evaluate_basic(board);
    return (board->side_to_move == WHITE) ? (basic_eval_white_view_pawn_units * 100.0f) : (-basic_eval_white_view_pawn_units * 100.0f);
#else
    // Fall back to basic evaluation if ONNX Runtime is not available
    float basic_eval_white_view_pawn_units = evaluate_basic(board);
    return (board->side_to_move == WHITE) ? (basic_eval_white_view_pawn_units * 100.0f) : (-basic_eval_white_view_pawn_units * 100.0f);
#endif
}

// Test the neural evaluation
void test_neural_evaluation(const Board *board) {
#if HAVE_ONNXRUNTIME
    printf("Attempting Neural evaluation: %.3f centipawns\n", evaluate_neural(board));
#else
    printf("Neural evaluation skipped (ONNX Runtime not available).\n");
#endif
    // Assuming evaluate_position might use evaluate_neural or evaluate_basic
    // printf("Classical evaluation (evaluate_position): %.3f\n", evaluate_position(board));
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

// The public run_neural_inference is defined in neural.h and implemented here as a wrapper.
// This function is NOT static.
bool run_neural_inference(const float *input_tensor_values, float *output_value) {
    return run_neural_inference_impl(input_tensor_values, output_value);
}
