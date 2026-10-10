#!/bin/sh
# Build tools/dobutsu_solver.cpp -> external/dobutsu_solver (own code, no third-party dependency).
set -e
cd "$(dirname "$0")/.."
mkdir -p external
SDK=$(xcrun --show-sdk-path 2>/dev/null || true)
INC=""
[ -n "$SDK" ] && [ -d "$SDK/usr/include/c++/v1" ] && INC="-isystem $SDK/usr/include/c++/v1"
clang++ -std=c++17 -O3 -DNDEBUG $INC -pthread tools/dobutsu_solver.cpp -o external/dobutsu_solver
echo built external/dobutsu_solver
