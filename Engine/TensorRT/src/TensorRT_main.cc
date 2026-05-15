#include "TensorRT_main.hpp"
#include "SortTracking.h"

#ifdef USE_TENSORRT

#include "../include/TensorRT.hpp"
#include <cuda_runtime.h>

static SORTTRACKING sorttracking;
static YOLOv8* yolov8 = nullptr;
static std::vector<Object> objs;

extern const std::vector<std::vector<unsigned int>> SKELETON;
extern const std::vector<std::vector<unsigned int>> KPS_COLORS;
extern const std::vector<std::vector<unsigned int>> LIMB_COLORS;

void trt_set_sort_config(const SORTTRACKING::SortTrackingConfig& sort_cfg,
                         const sort_kpt::KeypointFilterConfig& kpt_cfg)
{
    sorttracking.SetConfig(sort_cfg);
    sort_kpt::ConfigureGlobalKeypointFilter(kpt_cfg);
}

bool trt_init(const char* lanepose_model_path,
              char* classify_model_path,
              const char* icon_path,
              Config& config)
{
    (void)config;
    cudaSetDevice(0);

    yolov8 = new YOLOv8(lanepose_model_path);
    yolov8->make_pipe(true);

    classifydetector.classify_init(classify_model_path);

    if (IconManager::Load_Picture(icon_path) == false) {
        std::cerr << "Load Icon Picture Failed" << std::endl;
        return false;
    }

    return true;
}

std::vector<TrackingBox> trt_process_frame(const cv::Mat& frame,
                                           cv::Mat& output_frame,
                                           Config& config,
                                           bool draw_visuals,
                                           std::vector<Object>* raw_objects)
{
    std::vector<TrackingBox> TrackingResult;

    if (yolov8 == nullptr) {
        std::cerr << "TensorRT not initialized" << std::endl;
        return TrackingResult;
    }

    yolov8->copy_from_Mat(frame, config.size);
    yolov8->infer();

    yolov8->postprocess_pose(objs,
                             config.score_thres,
                             config.iou_thres,
                             config.topk,
                             config.num_labels);

    if (raw_objects != nullptr) {
        *raw_objects = objs;
    }

    TrackingResult = sorttracking.TrackingResult(objs);

    if (draw_visuals) {
        output_frame = frame;
        yolov8->draw_pose(frame,
                          output_frame,
                          TrackingResult,
                          SKELETON,
                          KPS_COLORS,
                          LIMB_COLORS,
                          config.num_keypoint);
    } else {
        output_frame = frame.clone();
    }

    return TrackingResult;
}

#else

void trt_set_sort_config(const SORTTRACKING::SortTrackingConfig&, const sort_kpt::KeypointFilterConfig&) {}

bool trt_init(const char*, char*, const char*, Config&)
{
    std::cerr << "TensorRT not enabled" << std::endl;
    return false;
}

std::vector<TrackingBox> trt_process_frame(const cv::Mat&, cv::Mat&, Config&, bool, std::vector<Object>*)
{
    return {};
}

#endif
