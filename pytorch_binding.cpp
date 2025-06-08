#include "pytorch_binding.h"
#include "neural.h"  // For board_to_planes function
#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/mps.h>
#include <iostream>
#include <memory>
#include <vector>

// Global state for PyTorch model
static std::shared_ptr<torch::jit::script::Module> model = nullptr;
static bool is_initialized = false;
static torch::Device device = torch::kCPU;
static bool mps_available = false;

extern "C" {

bool initialize_pytorch(const char *model_path) {
    if (is_initialized && model) {
        std::cout << "PyTorch model already initialized." << std::endl;
        return true;
    }

    std::cout << "Initializing PyTorch model from: " << model_path << std::endl;

    try {
        // Check available devices with better error handling
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available, using GPU" << std::endl;
            device = torch::Device(torch::kCUDA);
        } else {
            // Check for MPS with proper initialization
            mps_available = torch::hasMPS();
            if (mps_available) {
                std::cout << "MPS (Metal) is available, attempting to use Apple GPU" << std::endl;
                try {
                    // Explicitly initialize MPS
                    torch::mps::synchronize();
                    device = torch::Device(torch::kMPS);
                } catch (const c10::Error &e) {
                    std::cerr << "MPS initialization failed: " << e.what() << std::endl;
                    std::cerr << "Falling back to CPU" << std::endl;
                    device = torch::Device(torch::kCPU);
                    mps_available = false;
                }
            } else {
                std::cout << "Using CPU (no GPU acceleration available)" << std::endl;
                device = torch::Device(torch::kCPU);
            }
        }

        // Load the model
        model = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));
        model->to(device);
        model->eval();

        is_initialized = true;
        std::cout << "PyTorch model initialized successfully on "
                  << (device.is_cuda() ? "CUDA" : (device.is_mps() ? "MPS" : "CPU"))
                  << std::endl;
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

        // Get scalar value (assuming model outputs a single value)
        float eval_value = output.item<float>();

        // Convert to centipawns and player perspective
        float centipawns = eval_value * 100.0f;

        // Return the score from current player's perspective
        // The model gives evaluation from white's perspective
        return (board->side_to_move == WHITE) ? centipawns : -centipawns;
    } catch (const c10::Error &e) {
        std::cerr << "Error running PyTorch inference: " << e.what() << std::endl;

        // If using MPS and getting an error, try to fallback to CPU for this evaluation
        if (device.is_mps()) {
            try {
                std::cerr << "Attempting fallback to CPU for this evaluation..." << std::endl;

                // Create input tensor on CPU
                // const int BOARD_SIZE = 8;
                // const int INPUT_CHANNELS = 14;
                const size_t tensor_size = BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS;

                std::vector<float> tensor_data(tensor_size);
                if (!board_to_planes(board, tensor_data.data(), tensor_size * sizeof(float))) {
                    return 0.0f;
                }

                auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
                torch::Tensor cpu_tensor = torch::from_blob(
                                               tensor_data.data(),
                                               {1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE},
                                               cpu_options)
                                               .clone();

                // Get a CPU copy of the model for this evaluation
                auto cpu_model = model->clone();
                cpu_model.to(torch::kCPU);

                // Run inference on CPU
                torch::NoGradGuard no_grad;
                std::vector<torch::jit::IValue> cpu_inputs;
                cpu_inputs.push_back(cpu_tensor);

                torch::Tensor cpu_output = cpu_model.forward(cpu_inputs).toTensor();
                float eval_value = cpu_output.item<float>();
                float centipawns = eval_value * 100.0f;

                return (board->side_to_move == WHITE) ? centipawns : -centipawns;
            } catch (const c10::Error &cpu_err) {
                std::cerr << "CPU fallback also failed: " << cpu_err.what() << std::endl;
                return 0.0f;
            }
        }

        return 0.0f;
    }
}

bool is_pytorch_initialized(void) {
    return is_initialized && model != nullptr;
}

}  // extern "C"