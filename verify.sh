#!/usr/bin/env bash
#
# verify.sh - compile the Java standard-library pruning budget searcher and
# actually run its end-to-end test suite on a Linux CPU.
#
# No model downloads and no external services are used. Exits non-zero when
# compilation fails, any test fails, or when the suite discovers zero test
# cases (an empty/zero-case suite must fail rather than pass vacuously).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MAIN_SRC_DIR="src/main/java"
TEST_SRC_DIR="src/test/java"
OUT_DIR="build/classes"

if ! command -v javac >/dev/null 2>&1; then
    echo "ERROR: javac was not found on PATH (install a JDK)." >&2
    exit 1
fi
if ! command -v java >/dev/null 2>&1; then
    echo "ERROR: java was not found on PATH (install a JRE/JDK)." >&2
    exit 1
fi

echo "javac: $(javac -version 2>&1)"
echo "java:  $(java -version 2>&1 | head -n 1)"

mkdir -p "${OUT_DIR}"

echo "==> Compiling main sources"
find "${MAIN_SRC_DIR}" -name '*.java' > build/main-sources.txt
if [ ! -s build/main-sources.txt ]; then
    echo "ERROR: no main sources found under ${MAIN_SRC_DIR}" >&2
    exit 1
fi
javac -d "${OUT_DIR}" @build/main-sources.txt

echo "==> Compiling test sources"
find "${TEST_SRC_DIR}" -name '*.java' > build/test-sources.txt
if [ ! -s build/test-sources.txt ]; then
    echo "ERROR: no test sources found under ${TEST_SRC_DIR}" >&2
    exit 1
fi
javac -cp "${OUT_DIR}" -d "${OUT_DIR}" @build/test-sources.txt

echo "==> Running end-to-end test suite"
set +e
java -cp "${OUT_DIR}" pruner.PrunerTests | tee build/test-output.txt
TEST_STATUS="${PIPESTATUS[0]}"
set -e

# Belt-and-suspenders: even if the runner ever exited 0 with no tests,
# treat a missing or zero-count summary line as failure.
SUMMARY_LINE="$(grep -E '^tests: ' build/test-output.txt | tail -n 1 || true)"
if [ -z "${SUMMARY_LINE}" ]; then
    echo "ERROR: test summary line was not produced." >&2
    exit 1
fi
TEST_TOTAL="$(printf '%s\n' "${SUMMARY_LINE}" | sed -E 's/^tests: ([0-9]+),.*/\1/')"
if [ -z "${TEST_TOTAL}" ] || [ "${TEST_TOTAL}" -le 0 ]; then
    echo "ERROR: zero test cases discovered; an empty suite must fail." >&2
    exit 1
fi

if [ "${TEST_STATUS}" -ne 0 ]; then
    echo "ERROR: test suite failed (exit ${TEST_STATUS})." >&2
    exit "${TEST_STATUS}"
fi

echo "==> All ${TEST_TOTAL} tests passed."
