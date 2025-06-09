#ifndef SELF_PLAY_H
#define SELF_PLAY_H

#include <stdbool.h>

// Generate and export self-play games to a JSON file
bool generate_self_play_games(const char *model_path, const char *output_path,
                              int num_games, float temperature);

#endif  // SELF_PLAY_H
