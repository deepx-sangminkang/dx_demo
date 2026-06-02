# Vendored dx_stream Build — Design

Date: 2026-06-02

## Problem

The `dxstream` inference backend requires the external `dx_stream` repo to be
cloned and built/installed separately (`scripts/install_dxstream.sh` locates or
clones `dx_stream`, then runs its `install.sh` + `build.sh`). This is a hard,
out-of-tree dependency: users must obtain and build `dx_stream` before the demo
works.

Goal: let users build everything they need **from inside `yolo26_4ch_demo`**, with
no separate `dx_stream` checkout.

## What the demo actually uses

GStreamer elements: `dxpreprocess`, `dxinfer`, `dxpostprocess`, `dxscale`.
Python bindings: `pydxs` (`dx_get_frame_meta` → `DXFrameMeta`/`DXObjectMeta`).

The upstream plugin is a single `libgstdxstream.so` that also registers many
unused elements (`dxosd`, `dxtracker`, `dxgather`, `dxrate`, `dxconvert`,
`dxinputselector`, `dxoutputselector`, `dxmsgconv`, `dxmsgbroker`), pulling heavy
deps (`rdkafka`, `libmosquitto`, `eigen3`).

## Approach (chosen)

Vendor **only the source closure** needed for the 4 elements + pydxs into
`third_party/dxstream/`, with a trimmed meson build that:
- compiles only the 4 elements + metadata + preprocessors + transforms,
- registers only the 4 elements,
- drops `eigen3`, `rdkafka`, `libmosquitto` deps.

`librga` stays optional (RK3588), with libyuv fallback (so it also builds on x86
dev hosts for CI/verification).

### Vendored file closure (traced from includes)

```
third_party/dxstream/
  release.ver
  gst-dxstream-plugin/
    meson.build            (trimmed root)
    meson_options.txt
    general/dxcommon.hpp
    metadata/gst-dxframemeta.{hpp,cpp}
    metadata/gst-dxobjectmeta.{hpp,cpp}
    metadata/gst-dxusermeta.{hpp,cpp}
    src/
      meson.build          (trimmed)
      gst-dxstream.cpp     (trimmed registration: 4 elements only)
      gst-dxpreprocess.{hpp,cpp}
      gst-dxinfer.{hpp,cpp}
      gst-dxpostprocess.{hpp,cpp}
      gst-dxscale.{hpp,cpp}
      preprocessors/*
      transforms/{video_transform_factory,transform_kernel_base,
                  libyuv_transform_kernel,rga_transform_kernel,
                  video_transform_kernel.hpp,gst_frame_desc.hpp}
  bindings/python/pydxs/*
```

### Build script

`scripts/build_vendored_dxstream.sh` (meson setup/compile/install into a prefix,
then `pip install` pydxs with `PROJECT_ROOT` pointing at the vendored tree),
mirroring upstream `build.sh` but minimal. Writes `scripts/.dxstream_env.sh`
(already sourced by `run_demo.sh`).

`scripts/install_dxstream.sh` is updated to prefer the vendored tree and call the
new script, so no clone/locate of an external `dx_stream` is needed.

## Non-goals / unavoidable deps

DX-RT (`libdxrt`), GStreamer, OpenCV, json-glib, libyuv (and librga on RK3588)
remain system prerequisites — they are runtime/link deps of the C++ elements, not
of `dx_stream` packaging.

## Verification

Build the vendored plugin on the x86 dev host (librga absent → libyuv path) and
confirm `libgstdxstream.so` builds and `gst-inspect` lists the 4 elements; build
pydxs and import it.
