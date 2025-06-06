#ifndef NEURAL_H
#define NEURAL_H

#include "board.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h> // Add this for size_t definition

// Constants for the input tensor
#define BOARD_SIZE 8
#define INPUT_CHANNELS 14  // 12 piece planes + 2 metadata planes (side to move, en passant)

// Determine if ONNX Runtime is available (set by CMake)
#ifndef HAVE_ONNXRUNTIME
#define HAVE_ONNXRUNTIME 0
#endif

// Neural evaluator structure
typedef struct NeuralEvaluator NeuralEvaluator;

// Function to convert board to neural network input format
void board_to_planes(const Board* board, float* tensor_buffer, size_t buffer_size);

// Function to print the tensor representation for debugging
void print_tensor_representation(const Board* board);

// ONNX model functions
NeuralEvaluator* load_neural_evaluator(const char* model_path);
void free_neural_evaluator(NeuralEvaluator* evaluator);
float neural_evaluate_position(NeuralEvaluator* evaluator, const Board* board);

// Set/get the neural evaluator to use for evaluation
void set_neural_evaluator(NeuralEvaluator* evaluator);
NeuralEvaluator* get_neural_evaluator(void);

// Check if neural network support is available
bool is_neural_available(void);

#endif // NEURAL_H
