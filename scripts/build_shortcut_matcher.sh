#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/code/shortcut_pipeline/find_shortcut"
BUILD_DIR="${BUILD_DIR:-${SRC_DIR}/build_gcc12}"
BIN_DIR="${REPO_ROOT}/code/shortcut_pipeline/bin"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -f "${SRC_DIR}/CMakeLists.txt" ]]; then
  echo "[missing] matcher source tree: ${SRC_DIR}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${BIN_DIR}"

echo "[build] source: ${SRC_DIR}"
echo "[build] build dir: ${BUILD_DIR}"
echo "[build] output bin dir: ${BIN_DIR}"
echo "[build] jobs: ${JOBS}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j "${JOBS}"

cp "${BUILD_DIR}/cuda" "${BIN_DIR}/cuda"
chmod +x "${BIN_DIR}/cuda"

echo "[ok] wrote matcher binary to ${BIN_DIR}/cuda"
