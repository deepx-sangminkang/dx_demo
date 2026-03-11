#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <dxrt/dxrt_api.h>
#include "yolov8_seg_postprocess.h"

// forward declaration from seg_overlay.cpp
void bind_seg_overlay(pybind11::module_& m);

namespace py = pybind11;

dxrt::TensorPtrs numpy_to_dxrt_tensors(py::list ie_output) {
    const size_t num_tensors = ie_output.size();
    dxrt::TensorPtrs tensors;
    tensors.reserve(num_tensors);
    
    for (size_t i = 0; i < num_tensors; ++i) {
        py::array output_arr = py::cast<py::array>(ie_output[i]);
        py::buffer_info info = output_arr.request();
        
        dxrt::DataType dtype;
        if (info.format == py::format_descriptor<float>::format()) {
            dtype = dxrt::DataType::FLOAT;
        } else if (info.format == py::format_descriptor<int32_t>::format()) {
            dtype = dxrt::DataType::INT32;
        } else if (info.format == py::format_descriptor<int64_t>::format()) {
            dtype = dxrt::DataType::INT64;
        } else {
            throw std::runtime_error("Unsupported data type in numpy array");
        }
        
        const size_t num_dims = info.shape.size();
        std::vector<int64_t> shape;
        shape.reserve(num_dims);
        for (size_t j = 0; j < num_dims; ++j) {
            shape.emplace_back(static_cast<int64_t>(info.shape[j]));
        }
        
        auto tensor = std::make_shared<dxrt::Tensor>(
            "output_" + std::to_string(i),
            std::move(shape),
            dtype,
            info.ptr
        );
        
        tensors.emplace_back(std::move(tensor));
    }
    
    return tensors;
}

py::tuple yolov8_seg_results_to_numpy(const std::vector<YOLOv8_SEGResult>& results) {
    const size_t num_results = results.size();
    // detection array: [x1, y1, x2, y2, score, class_id]
    py::array_t<float> detections(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), 6});
    auto det_buf = detections.mutable_unchecked<2>();

    // mask array: (N, H, W) or (0, 0, 0) when empty
    if (num_results == 0) {
        py::array_t<uint8_t> empty_masks(std::vector<py::ssize_t>{0, 0, 0});
        return py::make_tuple(detections, empty_masks);
    }

    // Assume all results use the same mask size (postprocess fills them that way)
    int mask_h = results[0].mask_height;
    int mask_w = results[0].mask_width;
    py::array_t<uint8_t> masks(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(num_results), mask_h, mask_w});
    auto mask_buf = masks.mutable_unchecked<3>();

    for (size_t i = 0; i < num_results; ++i) {
        const auto& result = results[i];
        if (result.box.size() >= 4) {
            det_buf(i, 0) = result.box[0];
            det_buf(i, 1) = result.box[1];
            det_buf(i, 2) = result.box[2];
            det_buf(i, 3) = result.box[3];
        } else {
            det_buf(i, 0) = det_buf(i, 1) = det_buf(i, 2) = det_buf(i, 3) = 0.0f;
        }
        det_buf(i, 4) = result.confidence;
        det_buf(i, 5) = static_cast<float>(result.class_id);

        // Assume mask is a flat H*W array and convert it to uint8
        if (!result.mask.empty() && result.mask_height > 0 && result.mask_width > 0 &&
            static_cast<int>(result.mask.size()) == result.mask_height * result.mask_width) {
            for (int h = 0; h < result.mask_height; ++h) {
                for (int w = 0; w < result.mask_width; ++w) {
                    float v = result.mask[h * result.mask_width + w];
                    // Treat the value as 0/1 or a probability and scale to 0..255
                    if (v < 0.0f) v = 0.0f;
                    else if (v > 1.0f) v = 1.0f;
                    uint8_t mv = static_cast<uint8_t>(v * 255.0f);
                    mask_buf(i, h, w) = mv;
                }
            }
        } else {
            for (int h = 0; h < mask_h; ++h) {
                for (int w = 0; w < mask_w; ++w) {
                    mask_buf(i, h, w) = 0;
                }
            }
        }
    }

    return py::make_tuple(detections, masks);
}

PYBIND11_MODULE(dx_postprocess, m)
{
    py::class_<YOLOv8_SEGPostProcess>(m, "YOLOv8SegPostProcess")
        .def(py::init<int, int, float, float, bool>(),
             py::arg("input_w"),
             py::arg("input_h"),
             py::arg("score_threshold"),
             py::arg("nms_threshold"),
             py::arg("is_ort_configured"))
        .def("postprocess", [](YOLOv8_SEGPostProcess& self, py::list ie_output) {
            auto tensors = numpy_to_dxrt_tensors(ie_output);

            std::vector<YOLOv8_SEGResult> results;
            {
                py::gil_scoped_release release;
                results = self.postprocess(tensors);
            }
            
            return yolov8_seg_results_to_numpy(results);
        }, py::arg("ie_output"));

    // segmentation overlay utilities
    bind_seg_overlay(m);
}