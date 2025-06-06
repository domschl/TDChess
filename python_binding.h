#ifndef PYTHON_BINDING_H
#define PYTHON_BINDING_H

#include "board.h"
#include <stdio.h>
#include <stdbool.h>

/**
 * Export board positions to a dataset file for neural network training
 *
 * @param filename The name of the file to export to
 * @param positions Array of board positions
 * @param evaluations Array of evaluation scores for each position
 * @param count Number of positions to export
 * @return true if export was successful, false otherwise
 */
bool export_positions_to_dataset(const char *filename, const Board *positions,
                                 float *evaluations, size_t count);

/**
 * Export a single board to a JSON string
 *
 * @param board The board to export
 * @param buffer Buffer to store the JSON string
 * @param buffer_size Size of the buffer
 * @return true if export was successful, false otherwise
 */
bool export_board_to_json(const Board *board, char *buffer, size_t buffer_size);

#endif  // PYTHON_BINDING_H
