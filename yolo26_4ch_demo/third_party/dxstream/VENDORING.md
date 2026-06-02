# Vendored dx_stream subset

This directory contains a trimmed copy of [dx_stream](https://github.com/DEEPX-AI/dx_stream)
sources, vendored so the `dxstream` backend can be built in-tree (see
`../../docs/plans/2026-06-02-vendored-dxstream-design.md`).

- Upstream: https://github.com/DEEPX-AI/dx_stream
- release.ver: v3.0.1
- source commit: 0877d37
- License: LGPL (see ./LICENSE)

Only the elements the demo uses are included and built:
`dxpreprocess`, `dxinfer`, `dxpostprocess`, `dxscale`, plus the metadata the
`pydxs` bindings read. The unused elements (osd, tracker, gather, rate, convert,
selectors, msgconv/msgbroker) and their deps (rdkafka, libmosquitto, eigen3) are
omitted. `gst-dxstream-plugin/meson.build`, `src/meson.build` and
`src/gst-dxstream.cpp` are demo-local trimmed versions; all other files are
verbatim from upstream.

To refresh from a newer dx_stream, re-copy the listed files and re-apply the
trim in the three demo-local files above.
