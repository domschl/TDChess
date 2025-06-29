#ifndef PERFT_H
#define PERFT_H

#include "board.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Structure to hold PERFT test data
typedef struct {
    const char *name;
    const char *fen;
    unsigned long perftcnt[10];  // Array for counts at different depths
    int depth_count;             // Number of depths stored
} PerftData;

// Function to run PERFT tests
void run_perft_tests(int max_depth);
void test_perft(int max_depth);
void perft_detail(Board *board, int depth);
uint64_t perft(Board *board, int depth);

#ifdef __cplusplus
}
#endif

#endif  // PERFT_H
