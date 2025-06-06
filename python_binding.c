#include "python_binding.h"
#include "neural.h"
#include <string.h>
#include <stdlib.h>

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

    // Write each position
    for (size_t i = 0; i < count; i++) {
        char json_buffer[2048];
        if (!export_board_to_json(&positions[i], json_buffer, sizeof(json_buffer))) {
            fclose(file);
            return false;
        }

        fprintf(file, "    {\n");
        fprintf(file, "      \"board\": %s,\n", json_buffer);
        fprintf(file, "      \"evaluation\": %.6f%s\n",
                evaluations[i],
                (i < count - 1) ? "}," : "}");
    }

    // Write footer
    fprintf(file, "  ]\n");
    fprintf(file, "}\n");

    fclose(file);
    return true;
}

// Export a single board to a JSON string
bool export_board_to_json(const Board *board, char *buffer, size_t buffer_size) {
    if (!buffer || buffer_size < 1024) {
        return false;
    }

    // Allocate memory for tensor data
    size_t tensor_size = BOARD_SIZE * BOARD_SIZE * INPUT_CHANNELS;
    float *tensor = (float *)malloc(tensor_size * sizeof(float));
    if (!tensor) {
        return false;
    }

    // Convert board to tensor format
    board_to_planes(board, tensor, tensor_size * sizeof(float));

    // Build JSON representation
    int offset = 0;
    offset += snprintf(buffer + offset, buffer_size - offset, "{\n");
    offset += snprintf(buffer + offset, buffer_size - offset, "        \"tensor\": [");

    // Write tensor data as a flat array
    for (size_t i = 0; i < tensor_size; i++) {
        if (i > 0) {
            offset += snprintf(buffer + offset, buffer_size - offset, ", ");
        }
        offset += snprintf(buffer + offset, buffer_size - offset, "%.0f", tensor[i]);
    }

    offset += snprintf(buffer + offset, buffer_size - offset, "],\n");

    // Include FEN representation for readability
    char fen[128];
    board_to_fen(board, fen, sizeof(fen));
    offset += snprintf(buffer + offset, buffer_size - offset, "        \"fen\": \"%s\"\n", fen);

    offset += snprintf(buffer + offset, buffer_size - offset, "      }");

    free(tensor);
    return true;
}
