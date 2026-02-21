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

# Platform-specific PyTorch paths
if(APPLE)
    # macOS with homebrew
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "/opt/homebrew/opt/libtorch")
    message(STATUS "macOS detected: Using homebrew libtorch path")
elseif(UNIX AND NOT APPLE)
    # Linux - check for system-installed PyTorch (e.g., Fedora)
    find_path(TORCH_PYTHON_INSTALL_PATH
        NAMES torch
        PATHS /usr/lib64/python3.13/site-packages
              /usr/lib/python3.13/site-packages
              /usr/lib64/python3.12/site-packages
              /usr/lib/python3.12/site-packages
              /usr/lib64/python3.11/site-packages
              /usr/lib/python3.11/site-packages
        NO_DEFAULT_PATH
    )
    
    if(TORCH_PYTHON_INSTALL_PATH)
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "${TORCH_PYTHON_INSTALL_PATH}/torch")
        message(STATUS "Linux detected: Using system PyTorch at ${TORCH_PYTHON_INSTALL_PATH}/torch")
        
        # For system PyTorch, we need to find protobuf separately
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(PROTOBUF QUIET protobuf)
            if(PROTOBUF_FOUND)
                message(STATUS "Found protobuf via pkg-config: ${PROTOBUF_VERSION}")
                set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" "${PROTOBUF_PREFIX}")
            endif()
        endif()
        
        # Also try to find protobuf via standard CMake
        find_package(Protobuf QUIET)
        if(Protobuf_FOUND)
            message(STATUS "Found protobuf via CMake: ${Protobuf_VERSION}")
        endif()
    else()
        message(STATUS "Linux detected: System PyTorch not found, using default search paths")
    endif()
endif()

# Resolve protobuf conflicts by setting specific version requirements
# Only use static libs on macOS to avoid conflicts
if(APPLE)
    set(Protobuf_USE_STATIC_LIBS ON)
endif()

# Find PyTorch - handle protobuf conflicts
find_package(Torch REQUIRED)
message(STATUS "Found PyTorch: ${Torch_DIR}")

# Source files

set(SOURCES
    engine/main.c
    engine/board.c
    engine/movegen.c
    engine/eval.c
    engine/perft.c
    engine/search.c
    engine/neural.c
    engine/visualization.c
    engine/zobrist.c
    engine/python_binding.c
    engine/self_play.c
    engine/pytorch_binding.cpp
)


# PyTorch C++ binding is now in engine/
set_source_files_properties(engine/pytorch_binding.cpp PROPERTIES LANGUAGE CXX)


# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/engine)
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
