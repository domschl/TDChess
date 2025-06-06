cmake_minimum_required(VERSION 3.16)
project(TDChess C)

# Enable generation of compile_commands.json for VSCode IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C standard to C23
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find ONNX Runtime package
find_package(ONNXRuntime QUIET)

# Set a flag based on whether ONNX Runtime was found
if(ONNXRuntime_FOUND)
    add_compile_definitions(HAVE_ONNXRUNTIME=1)
    message(STATUS "ONNX Runtime found - enabling neural network support")
    
    # Check if we're on macOS and add Homebrew paths
    if(APPLE)
        include_directories(SYSTEM /opt/homebrew/include)
        link_directories(/opt/homebrew/lib)
        message(STATUS "Adding macOS Homebrew directories:")
        message(STATUS "  Include: /opt/homebrew/include")
        message(STATUS "  Library: /opt/homebrew/lib")
    endif()
else()
    add_compile_definitions(HAVE_ONNXRUNTIME=0)
    message(STATUS "ONNX Runtime not found - neural network support disabled")
endif()

# Define the TDChess executable
add_executable(TDChess
    main.c
    board.c
    movegen.c
    perft.c
    eval.c
    search.c
    neural.c
)

# Link with ONNX Runtime if available
if(ONNXRuntime_FOUND)
    # Option 1: Direct library linking approach
    if(APPLE)
        target_link_libraries(TDChess PRIVATE onnxruntime)
    else()
        target_link_libraries(TDChess PRIVATE ${ONNXRuntime_LIBRARIES})
    endif()
    
    # Additional linker flags if needed
    if(APPLE)
        # Add rpath for macOS to find the library at runtime
        set_target_properties(TDChess PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "/opt/homebrew/lib"
        )
    endif()
endif()

# Set C standard
set_target_properties(TDChess PROPERTIES
    C_STANDARD 23
    C_STANDARD_REQUIRED ON
)

# Add compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(TDChess PRIVATE -Wall -Wextra -Wpedantic -O3)
elseif(MSVC)
    target_compile_options(TDChess PRIVATE /W4 /O2)
endif()

# Install target
install(TARGETS TDChess DESTINATION bin)
