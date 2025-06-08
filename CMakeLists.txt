cmake_minimum_required(VERSION 3.16)
project(TDChess C CXX)

# Enable generation of compile_commands.json for VSCode IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C standard
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set C++ standard for PyTorch binding
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell CMake to find PyTorch first, before any other package that might use protobuf
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "/opt/homebrew/opt/libtorch")

# Resolve protobuf conflicts by setting specific version requirements
set(Protobuf_USE_STATIC_LIBS ON)

# Find PyTorch - handle protobuf conflicts
find_package(Torch REQUIRED)
message(STATUS "Found PyTorch: ${Torch_DIR}")

# Source files
set(SOURCES
    main.c
    board.c
    movegen.c
    eval.c
    perft.c
    search.c
    neural.c
    visualization.c
    td_learning.c
    python_binding.c
    self_play.c
)

# PyTorch C++ binding
set(SOURCES ${SOURCES} pytorch_binding.cpp)
set_source_files_properties(pytorch_binding.cpp PROPERTIES LANGUAGE CXX)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})

# Create executable
add_executable(TDChess ${SOURCES})

# Add GPU support option and definition
option(WITH_GPU_SUPPORT "Enable GPU support (CUDA/MPS) for PyTorch" ON)
if(WITH_GPU_SUPPORT)
    target_compile_definitions(TDChess PRIVATE USE_GPU_SUPPORT)
    message(STATUS "GPU support enabled for PyTorch.")
else()
    message(STATUS "GPU support disabled for PyTorch (CPU only).")
endif()

# Link libraries
target_link_libraries(TDChess ${TORCH_LIBRARIES} m)

# Silence PyTorch related deprecation warnings
target_compile_options(TDChess PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
)

# Installation
install(TARGETS TDChess DESTINATION bin)
