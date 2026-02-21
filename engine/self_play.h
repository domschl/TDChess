#ifndef SELF_PLAY_H
#define SELF_PLAY_H

#include <stdbool.h>
#include "board.h"
#include "movegen.h"

// Generate and export self-play games to a JSON file
bool generate_self_play_games(const char *model_path, const char *output_path,
                              int num_games, float temperature, unsigned int seed);

Move select_move_with_randomness(Board *board, float temperature);

#endif  // SELF_PLAY_H
