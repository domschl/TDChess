<!-- filepath: [copilot-instructions.md](http://_vscodecontentref_/5) -->
## Copilot Instructions for TDChess

### Project Overview
- This is a chess engine with neural network evaluation using PyTorch
- Primary language is C (C23 standard) with C++ for PyTorch bindings
- Uses CMake with Ninja build system
- Implements TD(λ) learning for training


### File Structure (Post-Refactor)
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

- `model/`: Stores datasets and trained models
- `build/`: CMake/Ninja build directory
- `CMakeLists.txt`: Build configuration (updated for new structure)

### Build System
- Uses CMake 3.16+ with Ninja
- Requires PyTorch C++ libraries (libtorch)
- Build directory: [build](http://_vscodecontentref_/9)
- Executable: [TDChess](http://_vscodecontentref_/10)

### Coding Conventions
- C23 standard for C code
- C++17 for PyTorch bindings
- Use snake_case for function and variable names
- Struct names use PascalCase
- Constants use UPPER_CASE
- Include comprehensive error handling
- Add proper documentation comments

### Neural Network Details
- Uses PyTorch for both training and inference
- Model format: TorchScript (.pt files)
- Input: 14 planes of 8×8 board representation
- Residual network architecture
- Handles both CUDA and MPS (Apple Silicon)
