#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "$0")")
DX_DEMO_PATH=$(realpath -s "${SCRIPT_DIR}/..")

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

# --- Initialize variables ---
ENABLE_DEBUG_LOGS=0   # New flag for debug logging
FORCE_ARGS=""

pushd $SCRIPT_DIR

# Function to display help message
show_help() {
    print_colored "Usage: $(basename "$0") [OPTIONS]" "YELLOW"
    print_colored "Options:" "GREEN"
    print_colored "  [-f|--force]                   Force overwrite if the file already exists" "GREEN"
    print_colored "  [-v|--verbose]                 Enable verbose (debug) logging." "GREEN"
    print_colored "  [-h|--help]                    Show this help message" "GREEN"

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
        -f|--force)
            FORCE_ARGS="--force"
            shift
            ;;
        -v|--verbose)
            ENABLE_DEBUG_LOGS=1
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            show_help "error" "Invalid option '$1'"
            ;;
    esac
done

print_colored "======== PATH INFO ========" "DEBUG"
print_colored "DX_DEMO_PATH($DX_DEMO_PATH)" "DEBUG"

setup_assets() {
    VIDEO_PATH=./assets/videos

    # Download model if not present
    print_colored "Checking model..." "INFO"
    ./scripts/setup_yolo11s-seg_model.sh || { print_colored "Model download failed." "ERROR"; exit 1; }

    SETUP_VIDEO_ARGS="--output=${VIDEO_PATH} --symlink_target_path=${DX_DEMO_PATH}/workspace/res/videos"

    print_colored "VIDEO_PATH: ${VIDEO_PATH}" "INFO"
    VIDEO_REAL_PATH=$(readlink -f "$VIDEO_PATH")
    # Check and set up videos
    if [ ! -d "$VIDEO_REAL_PATH" ] || [ "$FORCE_ARGS" != "" ]; then
        print_colored " Video directory not found. Running setup videos script... ($VIDEO_REAL_PATH)" "INFO"
        ./scripts/setup_sample_videos.sh $SETUP_VIDEO_ARGS $FORCE_ARGS || { print_colored "Setup videos script failed." "ERROR"; rm -rf $VIDEO_PATH; exit 1; }
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
