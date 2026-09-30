#!/usr/bin/env bash
# Compiles the pruning budget searcher and runs its automated test suite.
# Uses only the Java standard library. Fails when zero tests are present.
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

echo "==> Compiling sources and tests"
javac -d "$BUILD_DIR" $(find src test -name '*.java')

echo "==> Running tests"
java -cp "$BUILD_DIR" prune.Tests
