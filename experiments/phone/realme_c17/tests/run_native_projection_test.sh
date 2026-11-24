#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CXX="${CXX:-c++}"
"$CXX" -O3 -std=c++17 -Wall -Wextra -Werror \
  "$ROOT/experiments/phone/realme_c17/native_projection.cpp" \
  "$ROOT/experiments/phone/realme_c17/tests/native_projection_test.cpp" \
  -o "${TMPDIR:-/tmp}/qmini-realme-c17-projection-test"
"${TMPDIR:-/tmp}/qmini-realme-c17-projection-test"
