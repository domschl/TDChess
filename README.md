A chess engine with neural network evaluation and TD(λ) learning capabilities.

## Architecture

TDChess is a modular chess engine built in C (C23 standard) that combines traditional alpha-beta search with neural network evaluation trained using Temporal Difference (TD) learning.

### Core Components

- **Board Representation**: Uses a 64-square array with piece-color encoding for efficient state representation
- **Move Generation**: Legal move generator with bitboard-based attack detection
- **Evaluation**: 
  - Classical static evaluation function (material, piece positions)
  - Neural network evaluation using ONNX Runtime
- **Search**: 
  - Alpha-beta search with move ordering and transposition tables
  - Configurable search depth
- **Neural Network**: 
  - 14-channel input (6 piece types × 2 colors + 2 state planes)
  - Convolutional architecture with value head
  - ONNX model support for cross-platform compatibility
- **TD(λ) Learning**: Temporal difference learning with configurable λ parameter

### Neural Network Architecture

The neural network uses a CNN architecture specialized for chess position evaluation:

- **Input**: 14 planes of 8×8 boards
  - 12 piece planes (6 piece types × 2 colors)
  - 1 side-to-move plane
  - 1 en-passant plane
- **Convolutional Layers**:
  - First layer: 64 filters with 3×3 kernels
  - Second layer: 128 filters with 3×3 kernels
  - Third layer: 64 filters with 3×3 kernels
- **Value Head**:
  - Fully connected layers (4096 → 256 → 64 → 1)
  - Final tanh activation to produce evaluation in [-1, 1] range
  - Scaled to centipawns for traditional chess evaluation

## Building

TDChess uses CMake with Ninja as the build system.

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
```

### Training Commands

```
# Generate training dataset
./TDChess generate-dataset [file] [count] [depth]

# Run TD-Lambda training cycle
./TDChess td-lambda [initial_model] [output_model] [games] [lambda]
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
./TDChess generate-dataset chess_dataset.json 10000 3

# Train initial model
python train_neural.py --dataset chess_dataset.json --output chess_model.onnx --epochs 500 --batch-size 128

# Run TD-Lambda training
./TDChess td-lambda chess_model.onnx chess_model_improved.onnx 200 0.7

# Test the improved model
./TDChess play-neural chess_model_improved.onnx 4
```

or simply use `train_pipeline.sh`.

## Dependencies

- C compiler with C23 support
- CMake 3.12+
- Ninja build system
- UV for the python part

## TODOs

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
