cmake_minimum_required(VERSION 3.10)
project(TDChess C)

# C23 standard
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Use Ninja for builds
set(CMAKE_GENERATOR "Ninja")

# Sources
set(SOURCES 
    main.c
    board.c
    movegen.c
    perft.c
)

# Headers
set(HEADERS
    board.h
    movegen.h
    perft.h
)

# Define the executable
add_executable(TDChess ${SOURCES} ${HEADERS})

# Add compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(TDChess PRIVATE -Wall -Wextra -Wpedantic -O3)
elseif(MSVC)
    target_compile_options(TDChess PRIVATE /W4 /O2)
endif()

# Install target
install(TARGETS TDChess DESTINATION bin)
