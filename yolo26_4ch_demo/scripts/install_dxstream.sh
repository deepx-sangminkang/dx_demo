#!/bin/bash
# Compatibility entry point for the dxstream backend dependencies.
#
# The demo no longer requires a separate dx_stream checkout: the GStreamer
# plugin (dxpreprocess / dxinfer / dxpostprocess / dxscale) and the pydxs
# bindings are vendored under third_party/dxstream/ and built in-tree by
# build_vendored_dxstream.sh. This wrapper forwards to that script so existing
# callers (install.sh --with-dxstream, docs, CI) keep working.
#
# Usage:
#   scripts/install_dxstream.sh [--prefix=PATH] [--skip-deps] [--force] [--clean]

set -u
SCRIPT_DIR=$(realpath "$(dirname "$0")")
exec "${SCRIPT_DIR}/build_vendored_dxstream.sh" "$@"
