#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_DEMO_PATH=$(realpath -s "${SCRIPT_DIR}")

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

pushd ${DX_DEMO_PATH} >&2
DX_SRC_DIR=$PWD
target_arch=$(uname -m)

export DEBIAN_FRONTEND=noninteractive

function pkgs_installed() {
    for pkg in "$@"; do
        dpkg -s "$pkg" &>/dev/null || return 1
    done
    return 0
}

function compare_version() {
    awk -v n1="$1" -v n2="$2" 'BEGIN { if (n1 >= n2) exit 0; else exit 1; }'
}

function install_dep() {
    cmake_version_required=3.14
    install_cmake=false

    local dep_pkgs=(build-essential make zlib1g-dev libcurl4-openssl-dev wget tar zip cmake)
    if ! pkgs_installed "${dep_pkgs[@]}"; then
        echo " Install dependence package tools "
        sudo apt-get update
        if [ $? -ne 0 ]; then
            echo "Failed to apt update."
            exit 1
        fi
        sudo apt-get -y install "${dep_pkgs[@]}"
    else
        echo " [SKIP] dep packages already installed"
    fi

    cmake_version=$(cmake --version |grep -oP "\d+\.\d+\.\d+")
    if compare_version "$cmake_version" "$cmake_version_required"; then
        install_cmake=false
    else
        install_cmake=true
    fi
    if [ "$install_cmake" == true ]; then
        if ! test -e $DX_SRC_DIR/util; then
            mkdir $DX_SRC_DIR/util
        fi
        cd $DX_SRC_DIR/util
        if ! test -e $DX_SRC_DIR/util/cmake-$cmake_version_required.0; then
            echo " Install CMake v$$cmake_version_required.0 "
            wget https://cmake.org/files/v$cmake_version_required/cmake-$cmake_version_required.0.tar.gz --no-check-certificate
            tar xvf cmake-$cmake_version_required.0.tar.gz
        else
            echo " Already Exist CMake "
        fi
        cd cmake-$cmake_version_required.0
        ./bootstrap --system-curl
        make -j $(($(nproc) / 2))
        sudo make install
    fi

    if ! pkgs_installed ninja-build; then
        sudo apt install ninja-build
    else
        echo " [SKIP] ninja-build already installed"
    fi

    if ! pkgs_installed gcc-aarch64-linux-gnu g++-aarch64-linux-gnu; then
        sudo apt-get -y install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    else
        echo " [SKIP] aarch64 cross-compiler already installed"
    fi
}

function install_opencv() {
    local opencv_pkgs=(libjpeg-dev libtiff5-dev ffmpeg libavcodec-dev libavformat-dev \
        libswscale-dev libxvidcore-dev libavutil-dev libtbb-dev libeigen3-dev libx264-dev \
        libv4l-dev v4l-utils)
    if ! pkgs_installed "${opencv_pkgs[@]}"; then
        echo " Install opencv dependent library "
        sudo apt-get update
        sudo apt -y install "${opencv_pkgs[@]}"
    else
        echo " [SKIP] opencv dependent packages already installed"
    fi

    if ! pkgs_installed libfreetype-dev; then
        if apt-cache show libfreetype-dev > /dev/null 2>&1; then
            sudo apt-get install -y libfreetype-dev
        fi
    else
        echo " [SKIP] libfreetype-dev already installed"
    fi

    if ! pkgs_installed libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev; then
        sudo apt-get clean && sudo apt update && sudo apt-get -y upgrade
        sudo apt -y install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
    else
        echo " [SKIP] gstreamer packages already installed"
    fi

    if ! pkgs_installed libgtk2.0-dev; then
        sudo apt-get clean && sudo apt update && sudo apt-get -y upgrade
        sudo apt-get -y install libgtk2.0-dev
    else
        echo " [SKIP] libgtk2.0-dev already installed"
    fi

    if ! pkgs_installed libopencv-dev; then
        sudo apt -y --reinstall install libopencv-dev python3-opencv
        if [ $? -ne 0 ]; then
            echo "Failed to install OpenCV dependent libraries."
            exit 1
        fi
    else
        echo " [SKIP] libopencv-dev already installed"
    fi
}

if [ "$target_arch" == "arm64" ]; then
    target_arch=aarch64
fi

install_dep
install_opencv

print_colored "--- Running setup.sh ---" "INFO"
${SCRIPT_DIR}/setup.sh || { print_colored "setup.sh failed." "ERROR"; exit 1; }

print_colored "--- Running build.sh ---" "INFO"
${SCRIPT_DIR}/build.sh --arch $target_arch || { print_colored "build.sh failed." "ERROR"; exit 1; }

popd >&2
