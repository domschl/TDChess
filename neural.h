#ifndef NEURAL_H
#define NEURAL_H

#include "board.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Constants for the input tensor
#define BOARD_SIZE 8
#define INPUT_CHANNELS 14  // 12 piece planes + 2 metadata planes (side to move, en passant)

#if HAVE_ONNXRUNTIME
#include <onnxruntime/onnxruntime_c_api.h>

// Neural network evaluator structure
typedef struct {
    const OrtApi *ort;
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *memory_info;
    const char *input_names[1];
    const char *output_names[1];
} NeuralEvaluator;

// Global evaluator singleton
NeuralEvaluator *get_neural_evaluator(void);
#endif

// Neural network interface functions
bool initialize_neural(const char *model_path);
void shutdown_neural(void);
float evaluate_neural(const Board *board);
bool board_to_planes(const Board *board, float *planes, size_t size);

// Testing and debugging functions
void test_neural_input(void);
void test_neural_evaluation(const Board *board);

#endif  // NEURAL_H
