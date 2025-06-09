#ifndef PYTORCH_BINDING_H
#define PYTORCH_BINDING_H

#include <stdbool.h>
#include "board.h"  // For Board struct

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the PyTorch neural network from a model file.
 * @param model_path Path to the saved PyTorch model (.pt file)
 * @return true if initialization was successful, false otherwise
 */
bool initialize_pytorch(const char *model_path);

/**
 * Shut down the PyTorch neural network and free resources.
 */
void shutdown_pytorch(void);

/**
 * Evaluate a board position using the PyTorch neural network.
 * Returns score in CENTIPAWNS from the CURRENT PLAYER's perspective.
 * @param board The board position to evaluate
 * @return The evaluation score in centipawns
 */
float evaluate_pytorch(const Board *board);

/**
 * Check if PyTorch model is initialized and ready for inference.
 * @return true if PyTorch is initialized, false otherwise
 */
bool is_pytorch_initialized(void);

#ifdef __cplusplus
}
#endif

#endif  // PYTORCH_BINDING_H
