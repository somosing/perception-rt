#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="${VIRTUAL_ENV:-$PROJECT_ROOT/.venv}"
TRT_HEADERS_ROOT="$VENV_ROOT/tensorrt_cpp"
CUDA_HEADERS_ROOT="$VENV_ROOT/cuda_cpp"
BUILD_ROOT="$PROJECT_ROOT/build/native"
NVIDIA_REPOSITORY="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"

TRT_HEADERS_PACKAGE="libnvinfer-headers-dev"
TRT_HEADERS_VERSION="11.2.1.2-1+cuda13.3"
TRT_HEADERS_FILE="libnvinfer-headers-dev_11.2.1.2-1+cuda13.3_amd64.deb"

CUDART_HEADERS_PACKAGE="cuda-cudart-dev-13-0"
CUDART_HEADERS_VERSION="13.0.88-1"
CUDART_HEADERS_FILE="cuda-cudart-dev-13-0_13.0.88-1_amd64.deb"

CUDA_CRT_PACKAGE="cuda-crt-13-0"
CUDA_CRT_VERSION="13.0.88-1"
CUDA_CRT_FILE="cuda-crt-13-0_13.0.88-1_amd64.deb"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

extract_package() {
    local package="$1"
    local version="$2"
    local filename="$3"
    local destination="$4"
    local archive="$TEMPORARY_DIRECTORY/$filename"

    curl --fail --location \
        "$NVIDIA_REPOSITORY/$filename" \
        --output "$archive"

    if [[ "$(dpkg-deb -f "$archive" Package)" != "$package" ]]; then
        echo "Unexpected package name in $filename" >&2
        exit 1
    fi
    if [[ "$(dpkg-deb -f "$archive" Version)" != "$version" ]]; then
        echo "Unexpected package version in $filename" >&2
        exit 1
    fi
    if [[ "$(dpkg-deb -f "$archive" Architecture)" != "amd64" ]]; then
        echo "Unexpected package architecture in $filename" >&2
        exit 1
    fi

    dpkg-deb -x "$archive" "$destination"
}

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "This setup currently supports Linux x86_64 only" >&2
    exit 1
fi

for command in cmake curl dpkg-deb; do
    require_command "$command"
done

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
    echo "Python virtual environment not found: $VENV_ROOT" >&2
    exit 1
fi

SITE_PACKAGES="$("$VENV_ROOT/bin/python" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"

TRT_INCLUDE="$TRT_HEADERS_ROOT/usr/include/x86_64-linux-gnu"
CUDA_INCLUDE="$CUDA_HEADERS_ROOT/usr/local/cuda-13.0/targets/x86_64-linux/include"
TRT_LIBRARY="$SITE_PACKAGES/tensorrt_libs/libnvinfer.so.11"
CUDART_LIBRARY="$SITE_PACKAGES/nvidia/cu13/lib/libcudart.so.13"

TEMPORARY_DIRECTORY="$(mktemp -d)"
trap 'rm -rf -- "$TEMPORARY_DIRECTORY"' EXIT

if [[ ! -f "$TRT_INCLUDE/NvInferRuntime.h" ]]; then
    extract_package \
        "$TRT_HEADERS_PACKAGE" \
        "$TRT_HEADERS_VERSION" \
        "$TRT_HEADERS_FILE" \
        "$TRT_HEADERS_ROOT"
fi

if [[ ! -f "$CUDA_INCLUDE/cuda_runtime_api.h" ]]; then
    extract_package \
        "$CUDART_HEADERS_PACKAGE" \
        "$CUDART_HEADERS_VERSION" \
        "$CUDART_HEADERS_FILE" \
        "$CUDA_HEADERS_ROOT"
fi

if [[ ! -f "$CUDA_INCLUDE/crt/host_defines.h" ]]; then
    extract_package \
        "$CUDA_CRT_PACKAGE" \
        "$CUDA_CRT_VERSION" \
        "$CUDA_CRT_FILE" \
        "$CUDA_HEADERS_ROOT"
fi

for required_file in \
    "$TRT_INCLUDE/NvInferRuntime.h" \
    "$CUDA_INCLUDE/cuda_runtime_api.h" \
    "$CUDA_INCLUDE/crt/host_defines.h" \
    "$TRT_LIBRARY" \
    "$CUDART_LIBRARY"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required native dependency not found: $required_file" >&2
        exit 1
    fi
done

cmake \
    -S "$PROJECT_ROOT/cpp" \
    -B "$BUILD_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_INCLUDE_DIR="$TRT_INCLUDE" \
    -DTENSORRT_LIBRARY="$TRT_LIBRARY" \
    -DCUDART_INCLUDE_DIR="$CUDA_INCLUDE" \
    -DCUDART_LIBRARY="$CUDART_LIBRARY"

cmake --build "$BUILD_ROOT" --parallel
ctest --test-dir "$BUILD_ROOT" --output-on-failure

echo "Native executable: $BUILD_ROOT/perception_rt_native"
