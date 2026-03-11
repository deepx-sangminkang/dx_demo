#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_DEMO_PATH=$(realpath -s "${SCRIPT_DIR}/..")

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

# --- Initialize variables ---
ENABLE_DEBUG_LOGS=0   # New flag for debug logging
DOCKER_VOLUME_PATH=${DOCKER_VOLUME_PATH}
FORCE_ARGS=""
FORCE_REMOVE_VIDEOS=0

pushd $SCRIPT_DIR

# Function to display help message
show_help() {
    print_colored "Usage: $(basename "$0") [OPTIONS]" "YELLOW"
    print_colored "Options:" "GREEN"
    print_colored "  --docker_volume_path=<path>    Set Docker volume path (required in container mode)" "GREEN"
    print_colored "  [--force]                      Force overwrite if the file already exists" "GREEN"
    print_colored "  [--force-remove-videos]        Force remove videos if they exist" "GREEN"
    print_colored "  [--verbose]                    Enable verbose (debug) logging." "GREEN"
    print_colored "  [--help]                       Show this help message" "GREEN"

    if [ "$1" == "error" ] && [[ ! -n "$2" ]]; then
        print_colored "Invalid or missing arguments." "ERROR"
        exit 1
    elif [ "$1" == "error" ] && [[ -n "$2" ]]; then
        print_colored "$2" "ERROR"
        exit 1
    elif [[ "$1" == "warn" ]] && [[ -n "$2" ]]; then
        print_colored "$2" "WARNING"
        return 0
    fi
    exit 0
}

# Parse arguments
for i in "$@"; do
    case $1 in
        --docker_volume_path=*)
            DOCKER_VOLUME_PATH="${i#*=}"
            shift
            ;;
        --force)
            FORCE_ARGS="--force"
            shift
            ;;
        --force-remove-videos)
            FORCE_REMOVE_VIDEOS=1
            shift
            ;;
        --verbose)
            ENABLE_DEBUG_LOGS=1
            shift # Consume argument
            ;;
        --help)
            show_help
            ;;
        *)
            show_help "error" "Invalid option '$1'"
            ;;
    esac
done

print_colored "======== PATH INFO =========" "DEBUG"
print_colored "DX_DEMO_PATH($DX_DEMO_PATH)" "DEBUG"

# Default values
print_colored "=== DOCKER_VOLUME_PATH($DOCKER_VOLUME_PATH) is set ===" "INFO"

setup_assets() {
    VIDEO_PATH=./assets/videos
    CONTAINER_MODE=false

    # Check if running in a container
    if grep -qE "/docker|/lxc|/containerd" /proc/1/cgroup || [ -f /.dockerenv ]; then
        CONTAINER_MODE=true
        print_colored "(container mode detected)" "INFO"
        
        if [ -z "$DOCKER_VOLUME_PATH" ]; then
            show_help "error" "--docker_volume_path must be provided in container mode."
            exit 1
        fi

        SETUP_VIDEO_ARGS="--output=${VIDEO_PATH} --symlink_target_path=${DOCKER_VOLUME_PATH}/res/videos"
    else
        print_colored "(host mode detected)" "INFO"
        SETUP_VIDEO_ARGS="--output=${VIDEO_PATH} --symlink_target_path=${DX_DEMO_PATH}/workspace/res/videos"
    fi

    print_colored "VIDEO_PATH: ${VIDEO_PATH}" "INFO"
    VIDEO_REAL_PATH=$(readlink -f "$VIDEO_PATH")
    # Check and set up videos
    if [ ! -d "$VIDEO_REAL_PATH" ] || [ "$FORCE_ARGS" != "" ]; then
        if [ $FORCE_REMOVE_VIDEOS -eq 1 ]; then
            FORCE_ARGS="--force"
        fi
        print_colored " Video directory not found. Running setup videos script... ($VIDEO_REAL_PATH)" "INFO"
        ./setup_sample_videos.sh $SETUP_VIDEO_ARGS $FORCE_ARGS || { print_colored "Setup videos script failed." "ERROR"; rm -rf $VIDEO_PATH; exit 1; }
    else
        print_colored " Video directory found. ($VIDEO_REAL_PATH)" "INFO"
    fi

    print_colored "[OK] Sample videos setup complete" "INFO"
}

main() {
    setup_assets
}

main

popd
