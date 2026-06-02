#!/bin/bash

# Build script for cuda-webcam-filter
# Usage: ./build_all.sh [debug|release] [test]

# Default to release build
BUILD_TYPE="Release"
RUN_TESTS=0

# Parse arguments
for arg in "$@"; do
    if [ "$arg" == "debug" ]; then
        BUILD_TYPE="Debug"
    elif [ "$arg" == "release" ]; then
        BUILD_TYPE="Release"
    elif [ "$arg" == "test" ]; then
        RUN_TESTS=1
    fi
done

echo "Building with CMAKE_BUILD_TYPE=${BUILD_TYPE}"

# Create build directory if it doesn't exist
if [ ! -d "$(pwd)" ]; then
    echo "Please run this script from the build directory"
    exit 1
fi

# Configure with CMake
if [ $RUN_TESTS -eq 1 ]; then
    echo "Configuring with tests enabled"
    cmake .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DRUN_UNIT_TESTS=ON
else
    echo "Configuring without tests"
    cmake .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
fi

# Build the project
echo "Building the project..."
cmake --build . --config ${BUILD_TYPE} -- -j$(nproc)

# Run tests if requested
if [ $RUN_TESTS -eq 1 ]; then
    echo "Running tests..."
    ctest -C ${BUILD_TYPE} --output-on-failure
fi

echo "Build completed!"
