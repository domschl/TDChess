#include "pytorch_binding.h"
#include "neural.h"  // For board_to_planes function
#include <torch/script.h> // This should be always included for torch::jit::script::Module, torch::Device, etc.
#include <torch/utils.h> // For torch::NoGradGuard and other utilities
#ifdef USE_GPU_SUPPORT
#include <torch/cuda.h>
#include <torch/mps.h>
#endif
#include <iostream>
#include <memory>
#include <vector>

// Global state for PyTorch model
static std::shared_ptr<torch::jit::script::Module> model = nullptr;
static bool is_initialized = false;
static torch::Device device = torch::kCPU; // Default to CPU
#ifdef USE_GPU_SUPPORT
static bool mps_available = false; // Only relevant if USE_GPU_SUPPORT is defined
#endif

// Constants for evaluation scaling
static const float MAX_EVAL = 20.0f;

extern "C" {

bool initialize_pytorch(const char *model_path) {
    if (is_initialized && model) {
        std::cout << "PyTorch model already initialized." << std::endl;
        return true;
    }

    std::cout << "Initializing PyTorch model from: " << model_path << std::endl;

    try {
        // Device selection logic
    #ifdef USE_GPU_SUPPORT
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available, using GPU" << std::endl;
            device = torch::Device(torch::kCUDA);
        } else {
            mps_available = torch::hasMPS(); // torch::hasMPS() might still be callable without the full MPS header for linking
            if (mps_available) {
                std::cout << "MPS (Metal) is available, attempting to use Apple GPU" << std::endl;
                try {
                    torch::mps::synchronize(); // This call requires the MPS header and library symbols
                    device = torch::Device(torch::kMPS);
                } catch (const c10::Error &e) {
                    std::cerr << "MPS initialization failed: " << e.what() << std::endl;
                    std::cerr << "Falling back to CPU" << std::endl;
                    device = torch::Device(torch::kCPU);
                    mps_available = false; // Ensure this is reset
                }
            } else {
                std::cout << "CUDA and MPS not available, using CPU" << std::endl;
                device = torch::Device(torch::kCPU);
            }
        }
    #else // !USE_GPU_SUPPORT
        std::cout << "GPU support disabled, using CPU." << std::endl;
        device = torch::Device(torch::kCPU); // Explicitly set to CPU
    #endif // USE_GPU_SUPPORT

        // Load the model
        model = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));
        model->to(device); // Move model to the selected device
        model->eval();

        is_initialized = true;
    #ifdef USE_GPU_SUPPORT
        std::cout << "PyTorch model initialized successfully on "
                  << (device.is_cuda() ? "CUDA" : (mps_available && device.is_mps() ? "MPS" : "CPU"))
                  << std::endl;
    #else
        std::cout << "PyTorch model initialized successfully on CPU (GPU support disabled)." << std::endl;
    #endif
        return true;
    } catch (const c10::Error &e) {
        std::cerr << "Error initializing PyTorch model: " << e.what() << std::endl;
        model = nullptr;
        is_initialized = false;
        return false;
    }
}

void shutdown_pytorch(void) {
    if (is_initialized) {
        std::cout << "Shutting down PyTorch model" << std::endl;
        model = nullptr;
        is_initialized = false;
    }
}

float evaluate_pytorch(const Board *board) {
    if (!is_initialized || !model) {
        std::cerr << "PyTorch model not initialized" << std::endl;
        return 0.0f;
    }

    try {
        // Convert board to tensor format
        // const int BOARD_SIZE = 8;
        // const int INPUT_CHANNELS = 14;
        const size_t tensor_size = BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS;

        std::vector<float> tensor_data(tensor_size);

        // Use existing board_to_planes function to fill tensor_data
        if (!board_to_planes(board, tensor_data.data(), tensor_size * sizeof(float))) {
            std::cerr << "Failed to convert board to tensor format" << std::endl;
            return 0.0f;
        }

        // Create input tensor with CPU first, then move to target device
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input_tensor = torch::from_blob(
                                         tensor_data.data(),
                                         {1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE},
                                         options)
                                         .clone();

        // Move tensor to target device
        input_tensor = input_tensor.to(device);

        // Disable gradient calculation for inference
        torch::NoGradGuard no_grad;

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        torch::Tensor output = model->forward(inputs).toTensor();

        // Apply tanh activation to match training
        output = output.tanh();

        // Get scalar value (assuming model outputs a single value)
        float eval_value = output.item<float>();

        // Convert normalized [-1,1] output back to centipawns
        // First to pawn units by multiplying by MAX_EVAL
        float pawn_units = eval_value * MAX_EVAL;

        // Then to centipawns
        float centipawns = pawn_units * 100.0f;

        // Return the score from current player's perspective
        // The model gives evaluation from white's perspective
        return (board->side_to_move == WHITE) ? centipawns : -centipawns;
    } catch (const c10::Error &e) {
        std::cerr << "Error running PyTorch inference: " << e.what() << std::endl;

    #ifdef USE_GPU_SUPPORT
        // If using MPS and getting an error, try to fallback to CPU for this evaluation
        if (device.is_mps()) { // This check itself is fine, but the inner model.to(torch::kCPU) might be an issue if MPS symbols aren't linked.
                               // However, this code block is now only compiled if USE_GPU_SUPPORT is defined.
            try {
                std::cerr << "Attempting fallback to CPU for this evaluation..." << std::endl;

                // Create input tensor on CPU
                // const int BOARD_SIZE = 8;
                // const int INPUT_CHANNELS = 14;
                const size_t tensor_size = BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS;

                std::vector<float> tensor_data(tensor_size);
                if (!board_to_planes(board, tensor_data.data(), tensor_size * sizeof(float))) {
                    return 0.0f; // Ensure return on failure
                }

                auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
                torch::Tensor cpu_tensor = torch::from_blob(
                                               tensor_data.data(),
                                               {1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE},
                                               cpu_options)
                                               .clone();

                // Get a CPU copy of the model for this evaluation
                auto cpu_model = model->clone(); // clone() should be fine.
                cpu_model.to(torch::kCPU);      // to(CPU) is fine.

                // Run inference on CPU
                torch::NoGradGuard no_grad;
                std::vector<torch::jit::IValue> cpu_inputs;
                cpu_inputs.push_back(cpu_tensor);

                torch::Tensor cpu_output = cpu_model.forward(cpu_inputs).toTensor();
                // Apply tanh activation to match training (if it was also applied in main path before error)
                cpu_output = cpu_output.tanh();
                float eval_value = cpu_output.item<float>();

                // Convert normalized [-1,1] output back to centipawns
                float pawn_units = eval_value * MAX_EVAL;
                float centipawns = pawn_units * 100.0f;


                return (board->side_to_move == WHITE) ? centipawns : -centipawns;
            } catch (const c10::Error &cpu_err) {
                std::cerr << "CPU fallback also failed: " << cpu_err.what() << std::endl;
                return 0.0f; // Ensure return on failure
            }
        }
    #endif // USE_GPU_SUPPORT

        return 0.0f; // Default return if not MPS error or if USE_GPU_SUPPORT is not defined
    }
}

bool is_pytorch_initialized(void) {
    return is_initialized && model != nullptr;
}

}  // extern "C"