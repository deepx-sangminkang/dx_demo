#ifndef YOLOV11_SEG_POSTPROCESS_H
#define YOLOV11_SEG_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief YOLOv11 detection result structure
 * Contains bounding box coordinates, confidence scores, and class information
 */
struct YOLOv11_SEGResult {
    // Core detection data - using vectors for flexibility
    std::vector<float> box{};  // x1, y1, x2, y2 - bounding box coordinates
    float confidence{0.0f};    // Detection confidence score
    int class_id{0};           // Object class ID (0-79 for COCO classes)
    std::string class_name{};  // Object class name

    // Segmentation data
    std::vector<float> seg_mask_coef{};  // Segmentation mask coefficients (32 values)
    std::vector<float> mask{};           // Binary segmentation mask (flattened H*W)
    int mask_height{0};                  // Height of the segmentation mask
    int mask_width{0};                   // Width of the segmentation mask

    // Default constructor with explicit initialization
    YOLOv11_SEGResult() {}

    // Parameterized constructor with move semantics for better performance
    YOLOv11_SEGResult(std::vector<float> box_val, const float conf, const int cls_id,
                     const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    // Legacy constructor for backward compatibility
    YOLOv11_SEGResult(const std::vector<float>& box_val, const float conf, const int cls_id,
                     const std::string& cls_name);

    // Destructor
    ~YOLOv11_SEGResult() {}

    // Copy and move constructors/operators
    YOLOv11_SEGResult(const YOLOv11_SEGResult& other)
        : box(other.box),
          confidence(other.confidence),
          class_id(other.class_id),
          class_name(other.class_name),
          seg_mask_coef(other.seg_mask_coef),
          mask(other.mask),
          mask_height(other.mask_height),
          mask_width(other.mask_width) {}
    YOLOv11_SEGResult& operator=(const YOLOv11_SEGResult& other) {
        if (this != &other) {
            box = other.box;
            confidence = other.confidence;
            class_id = other.class_id;
            class_name = other.class_name;
            seg_mask_coef = other.seg_mask_coef;
            mask = other.mask;
            mask_height = other.mask_height;
            mask_width = other.mask_width;
        }
        return *this;
    }
    YOLOv11_SEGResult(YOLOv11_SEGResult&& other)
        : box(std::move(other.box)),
          confidence(other.confidence),
          class_id(other.class_id),
          class_name(std::move(other.class_name)),
          seg_mask_coef(std::move(other.seg_mask_coef)),
          mask(std::move(other.mask)),
          mask_height(other.mask_height),
          mask_width(other.mask_width) {}
    YOLOv11_SEGResult& operator=(YOLOv11_SEGResult&& other) {
        if (this != &other) {
            box = std::move(other.box);
            confidence = other.confidence;
            class_id = other.class_id;
            class_name = std::move(other.class_name);
            seg_mask_coef = std::move(other.seg_mask_coef);
            mask = std::move(other.mask);
            mask_height = other.mask_height;
            mask_width = other.mask_width;
        }
        return *this;
    }

    // Calculate area for NMS - const correctness
    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    // Validation methods
    bool is_invalid(int image_width, int image_height) const;
};

/**
 * @brief YOLOv11 post-processing class
 * Handles detection results processing, NMS, and coordinate transformations
 */
class YOLOv11_SEGPostProcess {
   private:
    // Image dimensions - using const for immutable values
    int input_width_{640};   // Model input width (default YOLO size)
    int input_height_{640};  // Model input height (default YOLO size)

    // Detection thresholds - using const for better performance
    float score_threshold_{0.5f};  // Class confidence threshold
    float nms_threshold_{0.45f};   // NMS IoU threshold

    // Model configuration - using const where appropriate
    enum { num_classes_ = 80 };  // Number of classes (COCO dataset)

    bool is_ort_configured_{false};  // Whether ORT inference is configured

    // Model-specific configuration parameters - using const where possible
    std::vector<std::string> cpu_output_names_;  // CPU output tensor names
    std::vector<std::string> npu_output_names_;  // NPU output tensor names (stride 8,16,32)
    std::map<int, std::vector<std::pair<int, int>>>
        anchors_by_strides_;  // Anchors organized by stride

    // Private helper methods - const correctness
    std::vector<YOLOv11_SEGResult> decoding_cpu_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOv11_SEGResult> decoding_npu_outputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<YOLOv11_SEGResult> apply_nms(const std::vector<YOLOv11_SEGResult>& detections) const;
    void decoding_mask_cpu_outputs(const dxrt::TensorPtrs& outputs,
                                   std::vector<YOLOv11_SEGResult>& detections);

    // Segmentation helper methods
    std::vector<std::vector<float>> process_segmentation_masks(
        const float* mask_output, const std::vector<YOLOv11_SEGResult>& detections, int mask_height,
        int mask_width) const;
    
    std::vector<std::vector<float>> scale_masks(
        std::vector<std::vector<float>>&& masks, int target_height, int target_width,
        int orig_height, int orig_width) const;
    
    std::vector<std::vector<float>> crop_masks(
        std::vector<std::vector<float>>&& masks,
        const std::vector<YOLOv11_SEGResult>& detections) const;
    

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

   public:
    /**
     * @brief Constructor with full configuration
     * @param input_w Model input width
     * @param input_h Model input height
     * @param score_threshold Class confidence threshold
     * @param nms_threshold NMS IoU threshold
     * @param is_ort_configured Whether ORT inference is configured (default:
     * false)
     * @note num_classes is fixed constant for COCO object detection
     */

    YOLOv11_SEGPostProcess(const int input_w, const int input_h, const float score_threshold,
                          const float nms_threshold, const bool is_ort_configured = false);

    YOLOv11_SEGPostProcess();

    /**
     * @brief Destructor
     */
    ~YOLOv11_SEGPostProcess() {}

    /**
     * @brief Process YOLOv11 model outputs
     * @param outputs Vector of output tensors from the model
     * @return Vector of processed detection results
     */
    std::vector<YOLOv11_SEGResult> postprocess(const dxrt::TensorPtrs& outputs);

    /**
     * @brief Align tensor data for processing
     * @param outputs Vector of output tensors from the model
     * @return Aligned tensor pointers
     */
    dxrt::TensorPtrs align_tensors(const dxrt::TensorPtrs& outputs) const;

    /**
     * @brief Set new thresholds
     * @param score_threshold New class confidence threshold
     * @param nms_threshold New NMS IoU threshold
     */
    void set_thresholds(const float score_threshold, const float nms_threshold);

    /**
     * @brief Get current configuration
     * @return String representation of current configuration
     */
    std::string get_config_info() const;

    // Getters for current configuration - const correctness
    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    float get_score_threshold() const { return score_threshold_; }
    float get_nms_threshold() const { return nms_threshold_; }
    bool get_is_ort_configured() const { return is_ort_configured_; }

    // Static configuration getters
    static int get_num_classes() { return num_classes_; }

    const std::map<int, std::vector<std::pair<int, int>>>& get_anchors_by_strides() const {
        return anchors_by_strides_;
    }

    // Model configuration getters
    const std::vector<std::string>& get_cpu_output_names() const { return cpu_output_names_; }
};

#endif  // YOLOV11_SEG_POSTPROCESS_H