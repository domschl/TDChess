#ifndef NEURAL_H
#define NEURAL_H

#include "board.h"
#include <stdbool.h>

// Input tensor dimensions
#define BOARD_SIZE 8
#define INPUT_CHANNELS 14  // 6 piece types * 2 colors + side to move + en passant

#ifdef __cplusplus
extern "C" {
#endif

// Function prototypes
bool initialize_neural(const char *model_path);
void shutdown_neural(void);
float evaluate_neural(const Board *board);
bool board_to_planes(const Board *board, float *tensor_buffer, size_t buffer_size_bytes);
void test_neural_evaluation(const Board *board);
void test_neural_input(void);

// Legacy function for compatibility
bool run_neural_inference(const float *input_tensor_values, float *output_value);

#ifdef __cplusplus
}
#endif

#endif  // NEURAL_H
