[![C/C++ CI](https://github.com/domschl/TDChess/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)

A chess engine with neural network evaluation and TD(λ) learning capabilities.

## Architecture

TDChess is a modular chess engine built in C (C23 standard) that combines traditional alpha-beta search with neural network evaluation trained using Temporal Difference (TD) learning.


## Directory Structure (Post-Refactor)

- `engine/`: All C/C++ source and header files for the chess engine
  - `main.c`: Entry point with command processing
  - `board.c/h`: Chess board representation
  - `movegen.c/h`: Move generation
  - `eval.c/h`: Classical evaluation
  - `neural.c/h`: Neural network interface
  - `pytorch_binding.cpp/h`: PyTorch C++ bindings
  - `search.c/h`: Alpha-beta search
  - `td_learning.c/h`: TD(λ) implementation
  - `self_play.c/h`: Self-play generation
- `training/`: All Python scripts for dataset generation, training, and pipeline
  - `train_neural.py`: Neural network training
  - `tdchess_pipeline.py`: TD(λ) training pipeline
  - `generate_stockfish_dataset.py`: Stockfish dataset generation
  - `diagnose_dataset.py`: Dataset analysis
  - `check_dataset.py`: Dataset integrity check
- `model/`: Stores datasets and trained models
- `build/`: CMake/Ninja build directory
- `CMakeLists.txt`: Build configuration (updated for new structure)

## Notes on Paths

- All Python scripts in `training/` expect paths relative to the project root (e.g., `../model/initial_dataset.json`, `../build/TDChess`).
- The CMake build system is updated to use sources from `engine/`.

## Usage

### Building the Engine

```sh
cmake -S . -B build -G Ninja
cmake --build build
```

### Running Training Scripts

```sh
cd training
python train_neural.py --dataset ../model/initial_dataset.json
python tdchess_pipeline.py --model-dir ../model
```

### Generating Datasets

```sh
cd training
python generate_stockfish_dataset.py
```
  - Scaled to centipawns for traditional chess evaluation

## Building

TDChess uses CMake with Ninja as the build system.

### Dependencies

- C compiler with C23 support
- C++ compiler with C++17 support (for PyTorch bindings)
- CMake 3.16+
- Ninja build system
- PyTorch C++ libraries (libtorch)

### Build Commands

```bash
mkdir build
cd build
cmake -G Ninja ..
ninja
```

## Usage

TDChess supports multiple modes of operation:

### Basic Commands

```
# Run perft (performance test) to count legal moves
./TDChess perft [depth]

# Show detailed perft results
./TDChess perft-detail [depth]

# Run perft tests on standard positions
./TDChess test [max_depth]

# Test evaluation function
./TDChess eval

# Play against the engine with classical evaluation
./TDChess play [depth]
```

### Neural Network Commands

```
# Test neural input representation
./TDChess neural

# Test neural evaluation with a trained model
./TDChess neural-eval [model]

# Play against the engine with neural evaluation
./TDChess play-neural [model] [depth]

# Test PyTorch device availability
./TDChess test-devices
```

### Training Commands

```
# Generate training dataset
./TDChess generate-dataset [file] [count] [depth]

# Generate self-play games
./TDChess generate-self-play [model.pt] [output.json] [games] [temperature]

# Run TD-Lambda training cycle
./TDChess td-lambda [initial_model.pt] [output_model.pt] [games] [lambda]

# Convert between model formats
./TDChess convert-model [input_model] [output_model.pt]
```

### Training pipeline

TDChess implements a complete neural network training pipeline:

1. Dataset Generation: Create training positions from self-play or existing games
2. Neural Training: Train the model using supervised learning on the dataset
3. TD(λ) Learning: Improve the model through temporal difference learning
4. Iterative Refinement: Continuously improve the model through self-play

Example training workflow:

```
# Generate initial dataset
./TDChess generate-dataset model/initial_dataset.json 20000 5

# Analyze dataset (optional)
python diagnose_dataset.py model/initial_dataset.json

# Run the complete training pipeline
python tdchess_pipeline.py

# Test the improved model
./TDChess play-neural model/chess_model_iter_50.pt 4
```

### Python Training Components

The Python-based training system includes:

- train_neural.py: Neural network training with PyTorch
- tdchess_pipeline.py: End-to-end training pipeline
- diagnose_dataset.py: Dataset analysis and validation

### PyTorch Integration

TDChess uses the PyTorch C++ API (libtorch) for neural network inference:

- Hardware Acceleration: Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU
- Model Format: Uses TorchScript models (.pt) for efficient deployment
- Fallback Mechanism: Automatically falls back to CPU if GPU inference fails

## Project Structure

- main.c - Main entry point
- board.c/h - Chess board representation and FEN parsing
- movegen.c/h - Move generation functionality
- eval.c/h - Classical evaluation functions
- neural.c/h - Neural network interface
- pytorch_binding.cpp/h - PyTorch C++ bindings
- search.c/h - Alpha-beta search implementation
- td_learning.c/h - Temporal difference learning
- self_play.c/h - Self-play game generation
- perft.c/h - Performance testing
- visualization.c/h - Board visualization

## License

TDChess is released under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
