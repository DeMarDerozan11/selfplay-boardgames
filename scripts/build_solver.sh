#!/bin/sh
# Clone and build Pascal Pons' Connect 4 solver (AGPL-3.0) into external/ (git-ignored).
# It is used only as an external program; none of its code is copied into this repo.
set -e
cd "$(dirname "$0")/.."
[ -d external/connect4-solver ] || git clone --depth 1 https://github.com/PascalPons/connect4 external/connect4-solver
cd external/connect4-solver
# Some Command Line Tools installs miss the libc++ include path; add the SDK one explicitly.
SDK=$(xcrun --show-sdk-path 2>/dev/null || true)
if [ -n "$SDK" ] && [ -d "$SDK/usr/include/c++/v1" ]; then
  make c4solver CXX="clang++ -isystem $SDK/usr/include/c++/v1"
else
  make c4solver
fi
