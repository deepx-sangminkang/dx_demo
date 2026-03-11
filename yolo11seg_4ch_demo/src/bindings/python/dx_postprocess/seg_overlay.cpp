#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>

namespace py = pybind11;

// Simple alpha blending helper: dst = (1-a)*dst + a*src
static inline uint8_t alpha_blend_u8(uint8_t dst, uint8_t src, float alpha) {
    float d = static_cast<float>(dst);
    float s = static_cast<float>(src);
    float out = (1.0f - alpha) * d + alpha * s;
    if (out < 0.0f) out = 0.0f;
    if (out > 255.0f) out = 255.0f;
    return static_cast<uint8_t>(out + 0.5f);
}

// image_bgr: (H, W, 3) uint8
// masks:     (N, H, W) uint8 (0 or 255)
// detections:(N, 6) float [x1,y1,x2,y2,score,class_id]
// palette:   (num_classes, 3) uint8
py::array_t<uint8_t> overlay_segmentation(
    py::array_t<uint8_t> image_bgr,
    py::array_t<uint8_t> masks,
    py::array_t<float> detections,
    py::array_t<uint8_t> palette,
    float alpha
) {
    // py::gil_scoped_release release;
    py::buffer_info img_info = image_bgr.request();
    if (img_info.ndim != 3)
        throw std::runtime_error("image_bgr must be HxWx3");

    const int H = static_cast<int>(img_info.shape[0]);
    const int W = static_cast<int>(img_info.shape[1]);
    const int C = static_cast<int>(img_info.shape[2]);
    if (C != 3)
        throw std::runtime_error("image_bgr must have 3 channels (BGR)");

    py::buffer_info mask_info = masks.request();
    if (mask_info.ndim != 3)
        throw std::runtime_error("masks must be NxHxW");

    const int N = static_cast<int>(mask_info.shape[0]);
    const int Hm = static_cast<int>(mask_info.shape[1]);
    const int Wm = static_cast<int>(mask_info.shape[2]);

    py::buffer_info det_info = detections.request();
    if (det_info.ndim != 2 || det_info.shape[1] < 6)
        throw std::runtime_error("detections must be Nx6");

    py::buffer_info pal_info = palette.request();
    if (pal_info.ndim != 2 || pal_info.shape[1] != 3)
        throw std::runtime_error("palette must be num_classes x 3");

    // Output image: copy of the original input
    py::array_t<uint8_t> out({img_info.shape[0], img_info.shape[1], img_info.shape[2]});
    py::buffer_info out_info = out.request();
    std::memcpy(out_info.ptr, img_info.ptr, img_info.size * sizeof(uint8_t));

    auto img_ptr = static_cast<uint8_t*>(out_info.ptr);
    auto mask_ptr = static_cast<uint8_t*>(mask_info.ptr);
    auto det_ptr = static_cast<float*>(det_info.ptr);
    auto pal_ptr = static_cast<uint8_t*>(pal_info.ptr);

    // Keep this simple: if the mask resolution differs from the image,
    // require Python to resize it before passing it in.
    if (Hm != H || Wm != W) {
        throw std::runtime_error("overlay_segmentation expects masks with same H,W as image");
    }

    const int img_stride_c = C;
    const int img_stride_w = C * W;

    {
        py::gil_scoped_release release;

        for (int i = 0; i < N; ++i) {
            const float x1_f = det_ptr[i * 6 + 0];
            const float y1_f = det_ptr[i * 6 + 1];
            const float x2_f = det_ptr[i * 6 + 2];
            const float y2_f = det_ptr[i * 6 + 3];
            const float score = det_ptr[i * 6 + 4];
            const int   cls   = static_cast<int>(det_ptr[i * 6 + 5]);

            if (score <= 0.0f) {
                continue;
            }

            const int cls_idx = std::max(0, std::min(cls, static_cast<int>(pal_info.shape[0]) - 1));
            const uint8_t b = pal_ptr[cls_idx * 3 + 0];
            const uint8_t g = pal_ptr[cls_idx * 3 + 1];
            const uint8_t r = pal_ptr[cls_idx * 3 + 2];

            const int x1 = std::max(0, std::min(static_cast<int>(x1_f), W - 1));
            const int y1 = std::max(0, std::min(static_cast<int>(y1_f), H - 1));
            const int x2 = std::max(0, std::min(static_cast<int>(x2_f), W - 1));
            const int y2 = std::max(0, std::min(static_cast<int>(y2_f), H - 1));

            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            const int mask_offset = i * H * W;

            for (int y = y1; y < y2; ++y) {
                const int row_offset_img = y * img_stride_w;
                const int row_offset_mask = mask_offset + y * W;

                for (int x = x1; x < x2; ++x) {
                    const uint8_t mv = mask_ptr[row_offset_mask + x];
                    if (mv == 0) {
                        continue;
                    }

                    const int idx = row_offset_img + x * img_stride_c;

                    img_ptr[idx + 0] = alpha_blend_u8(img_ptr[idx + 0], b, alpha);
                    img_ptr[idx + 1] = alpha_blend_u8(img_ptr[idx + 1], g, alpha);
                    img_ptr[idx + 2] = alpha_blend_u8(img_ptr[idx + 2], r, alpha);
                }
            }
        }
    }

    return out;
}

void bind_seg_overlay(py::module_& m) {
    m.def("overlay_segmentation", &overlay_segmentation,
          py::arg("image_bgr"),
          py::arg("masks"),
          py::arg("detections"),
          py::arg("palette"),
          py::arg("alpha") = 0.5f,
          R"doc(Apply segmentation masks as colored alpha-blended overlay on a BGR image.

image_bgr: HxWx3 uint8
masks:     NxHxW uint8 (0 or 255)
detections:Nx6 float [x1,y1,x2,y2,score,class_id]
palette:   num_classes x 3 uint8 (BGR colors)
alpha:     blending factor [0,1].
)doc");
}
