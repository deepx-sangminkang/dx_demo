common_args = [
  '-DPACKAGE_VERSION="' + project_version + '"',
  '-DPACKAGE="gst-dxstream"',
  '-DGST_LICENSE="LGPL"',
  '-DGST_PACKAGE_NAME="GStreamer DEEPX DX-Stream Plugins"',
  '-DGST_PACKAGE_ORIGIN="https://github.com/DEEPX-AI/dx_stream"'
]

if librga_flag
  common_args += '-DHAVE_LIBRGA'
endif

sources = [
    'gst-dxstream.cpp',
    'gst-dxpreprocess.cpp',
    'gst-dxinfer.cpp',
    'gst-dxpostprocess.cpp',
    'gst-dxscale.cpp',

    './../metadata/gst-dxframemeta.cpp',
    './../metadata/gst-dxobjectmeta.cpp',
    './../metadata/gst-dxusermeta.cpp',

    'preprocessors/preprocessor.cpp',
    'preprocessors/preprocessor_factory.cpp',

    'transforms/video_transform_factory.cpp',
    'transforms/transform_kernel_base.cpp',
    'transforms/libyuv_transform_kernel.cpp',
]

if librga_flag
  sources += [
    'transforms/rga_transform_kernel.cpp'
  ]
endif

# Install to GStreamer plugin directory under libdir
gst_plugin_dir = get_option('libdir') / 'gstreamer-1.0'

message('GStreamer plugin will be installed to: ' + get_option('prefix') / gst_plugin_dir)

# Build and install the shared library
gstdxstream = shared_library('gstdxstream',
    sources,
    cpp_args: common_args,
    include_directories: include_dirs,
    dependencies : extra_deps,
    install : true,
    install_dir : gst_plugin_dir,
    version: project_version,
    soversion: '0'
)

# Generate pkg-config file
pkg = import('pkgconfig')
pkg.generate(
    gstdxstream,
    name: 'gstdxstream',
    filebase: 'gstdxstream',
    description: 'DeepX Stream GStreamer Plugin',
    requires: ['gstreamer-1.0 >= 1.0'],
    variables: [
        'plugindir=' + '${libdir}' / 'gstreamer-1.0'
    ],
    install_dir: join_paths(get_option('prefix'), 'lib', 'pkgconfig')
)
