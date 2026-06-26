#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")

source "${SCRIPT_DIR}/scripts/color_env.sh"
source "${SCRIPT_DIR}/scripts/common_util.sh"

target_arch=$(uname -m)
build_type=Release
clean_build=false
verbose=false

help() {
    echo "Usage: $0 [OPTIONS]"
    echo "  --clean          Remove previous build artifacts"
    echo "  --verbose        Show detailed build commands"
    echo "  --type <TYPE>    CMake build type: Release|Debug|RelWithDebInfo (default: Release)"
    echo "  --arch <ARCH>    Target architecture: x86_64|aarch64 (default: $(uname -m))"
    echo "  --help           Show this help"
    exit 0
}

while (( $# )); do
    case "$1" in
        --help)    help;;
        --clean)   clean_build=true; shift;;
        --verbose) verbose=true; shift;;
        --type)    shift; build_type=$1; shift;;
        --arch)    shift; target_arch=$1; shift;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [ "$target_arch" == "arm64" ]; then
    target_arch=aarch64
fi

toolchain_file="${SCRIPT_DIR}/cmake/toolchain.${target_arch}.cmake"
if [ ! -f "$toolchain_file" ]; then
    echo -e "${TAG_ERROR} Toolchain file not found: $toolchain_file"
    exit 1
fi

build_dir="${SCRIPT_DIR}/build_${target_arch}"
bin_dir="${SCRIPT_DIR}/bin"

if [ "$clean_build" == "true" ]; then
    echo -e "${TAG_INFO} Cleaning build directories..."
    rm -rf "$build_dir" "$bin_dir"
fi

mkdir -p "$build_dir"

pushd "$build_dir" > /dev/null

cmake "${SCRIPT_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${toolchain_file}" \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}" \
    -DCMAKE_VERBOSE_MAKEFILE="${verbose}" \
    -G Ninja \
    || { echo -e "${TAG_ERROR} CMake configuration failed."; exit 1; }

cmake --build . --target install \
    || { echo -e "${TAG_ERROR} CMake build failed."; exit 1; }

popd > /dev/null

if [ -d "$bin_dir" ]; then
    echo -e "${TAG_INFO} Build done (${build_type}). Binary in: ${bin_dir}/"
    ls -1 "$bin_dir/"
else
    echo -e "${TAG_ERROR} Build failed — bin/ not created."
    exit 1
fi
