#include "python_binding.h"
#include "neural.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

// Export board positions to a dataset file for neural network training
bool export_positions_to_dataset(const char *filename, const Board *positions,
                                 float *evaluations, size_t count) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Failed to open file %s for writing\n", filename);
        return false;
    }

    // Write header
    fprintf(file, "{\n");
    fprintf(file, "  \"positions\": [\n");
    fflush(file);  // Flush after writing header

    // Write each position
    bool success = true;
    for (size_t i = 0; i < count; i++) {
        // Allocate a large enough buffer for the JSON representation
        // 8KB should be more than enough for a chess position
        char json_buffer[8192] = {0};

        if (!export_board_to_json(&positions[i], json_buffer, sizeof(json_buffer))) {
            printf("Failed to export board at index %zu to JSON\n", i);
            success = false;
            break;
        }

        fprintf(file, "    {\n");
        fprintf(file, "      \"board\": %s,\n", json_buffer);
        fprintf(file, "      \"evaluation\": %.6f%s\n",
                evaluations[i],
                (i < count - 1) ? "}," : "}");

        // Flush periodically to ensure data is written
        if (i % 10 == 0) {
            fflush(file);
        }
    }

    // Write footer
    fprintf(file, "  ]\n");
    fprintf(file, "}\n");
    fflush(file);  // Final flush before closing

    fclose(file);
    return success;
}

// Export a single board to a JSON string with improved buffer handling
bool export_board_to_json(const Board *board, char *buffer, size_t buffer_size) {
    if (!buffer || buffer_size < 2048) {
        printf("Buffer too small for JSON export\n");
        return false;
    }

    // Allocate memory for tensor data
    size_t tensor_size = BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS;
    float *tensor = (float *)malloc(tensor_size * sizeof(float));
    if (!tensor) {
        printf("Failed to allocate memory for tensor\n");
        return false;
    }

    // Convert board to tensor format
    board_to_planes(board, tensor, tensor_size * sizeof(float));

    // Build JSON representation
    size_t offset = 0;
    int i_written = snprintf(buffer + offset, buffer_size - offset, "{\n");
    if (i_written < 0 || (size_t)i_written >= buffer_size - offset) {
        printf("Buffer overflow in JSON generation\n");
        free(tensor);
        return false;
    }
    offset += (size_t)i_written;

    i_written = snprintf(buffer + offset, buffer_size - offset, "        \"tensor\": [");
    if (i_written < 0 || (size_t)i_written >= buffer_size - offset) {
        printf("Buffer overflow in JSON generation\n");
        free(tensor);
        return false;
    }
    offset += (size_t)i_written;

    // Write tensor data as a flat array
    for (size_t i = 0; i < tensor_size; i++) {
        if (i > 0) {
            i_written = snprintf(buffer + offset, buffer_size - offset, ", ");
            if (i_written < 0 || (size_t)i_written >= buffer_size - offset) {
                printf("Buffer overflow in tensor data\n");
                free(tensor);
                return false;
            }
            offset += (size_t)i_written;
        }

        i_written = snprintf(buffer + offset, buffer_size - offset, "%.0f", tensor[i]);
        if (i_written < 0 || (size_t)i_written >= buffer_size - offset) {
            printf("Buffer overflow in tensor data\n");
            free(tensor);
            return false;
        }
        offset += (size_t)i_written;
    }

    i_written = snprintf(buffer + offset, buffer_size - offset, "],\n");
    if (i_written < 0 || (size_t)i_written >= buffer_size - offset) {
        printf("Buffer overflow in JSON generation\n");
        free(tensor);
        return false;
    }
    offset += (size_t)i_written;

    // Include FEN representation for readability
    char fen[128];
    if (!board_to_fen(board, fen, sizeof(fen))) {
        printf("Failed to convert board to FEN\n");
        free(tensor);
        return false;
    }

    i_written = snprintf(buffer + offset, buffer_size - offset, "        \"fen\": \"%s\"\n", fen);
    if (i_written < 0 || (size_t)i_written >= buffer_size - offset) {
        printf("Buffer overflow in JSON generation\n");
        free(tensor);
        return false;
    }
    offset += (size_t)i_written;

    i_written = snprintf(buffer + offset, buffer_size - offset, "      }");
    if (i_written < 0 || (size_t)i_written >= buffer_size - offset) {
        printf("Buffer overflow in JSON generation\n");
        free(tensor);
        return false;
    }

    free(tensor);
    return true;
}
