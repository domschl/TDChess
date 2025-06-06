# TDChess

TDChess is a chess engine written in C that implements standard chess rules and provides performance benchmarking capabilities. The engine includes a move generator with comprehensive PERFT testing to verify correctness.

This project is entirely written by AI, demonstrating the capabilities of modern AI systems in creating functional chess engines.

## Features

- Standard chess rules implementation
- Efficient move generation
- Interactive mode for playing
- Performance testing (PERFT) for move generation
- Comprehensive test suite with standard chess positions

## Building from Source

### Prerequisites

- C compiler with C23 support
- CMake (3.14 or higher)
- Ninja build system

### Build Instructions

```bash
# Create a build directory
mkdir -p build && cd build

# Configure with CMake
cmake -G Ninja ..

# Build the project
ninja

# Alternatively, to build and run tests in one command
ninja && ./TDChess test 3
```

## Usage

TDChess provides several command-line options:

### Interactive Mode

```bash
./TDChess
```

This launches TDChess in interactive mode where you can play against the engine.

### PERFT Testing

```bash
# Run PERFT to a specific depth (e.g., 5)
./TDChess perft 5

# Show detailed PERFT results for a specific depth
./TDChess perft-detail 2

# Run a comprehensive suite of PERFT tests on standard positions
./TDChess test [max_depth]
```

PERFT (Performance Test) is a move generation testing technique that counts all possible legal moves to a specific depth. It's used to verify move generator correctness.

### Example PERFT Results

For the standard starting position:
- Depth 1: 20 moves
- Depth 2: 400 moves
- Depth 3: 8,902 moves
- Depth 4: 197,281 moves
- Depth 5: 4,865,609 moves

## Neural Evaluation Implementation Plan

TDChess will use a neural network for position evaluation. The implementation plan follows these steps:

### 1. Basic Evaluation Function (C)
- Implement a simple material-based evaluator in C
- Create the evaluation interface that will later support neural evaluation
- Test with simple gameplay mechanics

### 2. Board Representation for Neural Input (C)
- Create conversion functions from Board to tensor format
- Implement piece planes and auxiliary feature planes
- Design efficient memory layout for ONNX compatibility

### 3. ONNX Runtime Integration (C)
- Add ONNX Runtime as a dependency in CMake
- Create wrapper functions for model loading and inference
- Implement session management and memory handling

### 4. Python Bindings
- Create Python bindings for TDChess
- Expose board manipulation and move generation to Python
- Implement tensor conversion functions for PyTorch

### 5. Neural Network Architecture (Python/PyTorch)
- Implement CNN architecture in PyTorch
- Create model classes with appropriate input/output handling
- Set up testing infrastructure for model validation

### 6. Training Pipeline (Python)
- Create self-play generation code
- Implement position storage and batch creation
- Set up training loop with optimization

### 7. TD(λ) Learning Implementation (Python)
- Implement temporal difference learning logic
- Add eligibility traces for TD(λ)
- Create utilities for saving/loading models

### 8. ONNX Export (Python to C)
- Add export functionality to save PyTorch models as ONNX
- Verify compatibility with ONNX Runtime
- Create versioning system for models

### 9. Integration and Testing (C)
- Integrate neural evaluation into the main search
- Implement evaluation caching for performance
- Create evaluation benchmark tools

### 10. Tournament Play (C)
- Set up testing against other engines
- Implement UCI protocol for compatibility
- Create analysis mode for position evaluation

## Project Structure

- `main.c` - Main entry point
- `board.c/h` - Chess board representation and FEN parsing
- `movegen.c/h` - Move generation functionality
- `perft.c/h` - PERFT testing functionality

## License

TDChess is released under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
