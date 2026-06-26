#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "$0")")
RUNTIME_PATH=$(realpath -s "${SCRIPT_DIR}/../..")
DX_AS_PATH=$(realpath -s "${RUNTIME_PATH}/..")

# color env settings
source ${SCRIPT_DIR}/scripts/color_env.sh
source ${SCRIPT_DIR}/scripts/common_util.sh

# --- Initialize variables ---
ENABLE_DEBUG_LOGS=0   # New flag for debug logging
FORCE_ARGS=""
FORCE_REMOVE_MODELS=0
FORCE_REMOVE_VIDEOS=0
SYMLINK_TARGET_PATH="${SCRIPT_DIR}/../workspace"
SYMLINK_ARGS="--symlink_target_path=${SYMLINK_TARGET_PATH}"

pushd $SCRIPT_DIR

# Function to display help message
show_help() {
    print_colored "Usage: $(basename "$0") [OPTIONS]" "YELLOW"
    print_colored "Options:" "GREEN"
    print_colored "  [--force]                      Force overwrite if the file already exists" "GREEN"
    print_colored "  [--force-remove-models]        Force remove models if they exist" "GREEN"
    print_colored "  [--force-remove-videos]        Force remove videos if they exist" "GREEN"
    print_colored "  [--verbose]                    Enable verbose (debug) logging." "GREEN"
    print_colored "  [--symbolic-link-target-path=<path>]  Set symlink target path (default: ../workspace)" "GREEN"
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
        --force)
            FORCE_ARGS="--force"
            shift
            ;;
        --force-remove-models)
            FORCE_REMOVE_MODELS=1
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
        --symbolic-link-target-path=*)
            SYMLINK_TARGET_PATH="${1#*=}"
            SYMLINK_ARGS="--symlink_target_path=${SYMLINK_TARGET_PATH}"
            shift
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
print_colored "RUNTIME_PATH($RUNTIME_PATH)" "DEBUG"
print_colored "DX_AS_PATH($DX_AS_PATH)" "DEBUG"

# Default values

setup_assets() {
    MODEL_PATH=./assets/models
    VIDEO_PATH=./assets/videos
    SETUP_MODEL_ARGS="--output=${MODEL_PATH} ${SYMLINK_ARGS}"
    SETUP_VIDEO_ARGS="--output=${VIDEO_PATH} ${SYMLINK_ARGS}"

    print_colored " MODEL_PATH: ${MODEL_PATH}" "INFO"
    MODEL_REAL_PATH=$(readlink -f "$MODEL_PATH")
    # Check and set up models
    if [ ! -d "$MODEL_REAL_PATH" ] || [ "$FORCE_ARGS" != "" ]; then
        if [ $FORCE_REMOVE_MODELS -eq 1 ]; then
            FORCE_ARGS="--force"
        fi
        print_colored " models directory not found. Running setup models script... ($MODEL_REAL_PATH)" "INFO"
        ./scripts/setup_sample_models.sh $SETUP_MODEL_ARGS $FORCE_ARGS || { print_colored "Setup models script failed." "ERROR"; rm -rf $MODEL_PATH; exit 1; }
    else
        print_colored " models directory found. ($MODEL_REAL_PATH)" "INFO"
    fi

    print_colored "VIDEO_PATH: ${VIDEO_PATH}" "INFO"
    VIDEO_REAL_PATH=$(readlink -f "$VIDEO_PATH")
    # Check and set up models
    if [ ! -d "$VIDEO_REAL_PATH" ] || [ "$FORCE_ARGS" != "" ]; then
        if [ $FORCE_REMOVE_VIDEOS -eq 1 ]; then
            FORCE_ARGS="--force"
        fi
        print_colored " Video directory not found. Running setup models script... ($VIDEO_REAL_PATH)" "INFO"
        ./scripts/setup_sample_videos.sh $SETUP_VIDEO_ARGS $FORCE_ARGS || { print_colored "Setup videos script failed." "ERROR"; rm -rf $VIDEO_PATH; exit 1; }
    else
        print_colored " Video directory found. ($VIDEO_REAL_PATH)" "INFO"
    fi

    print_colored "[OK] Sample models and videos setup complete" "INFO"
}

main() {
    setup_assets
}

main

popd
