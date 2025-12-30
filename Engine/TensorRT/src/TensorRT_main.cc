#include "TensorRT_main.hpp"
#include "SortTracking.h"

#ifdef USE_TENSORRT

#include "../include/TensorRT.hpp"
#include <cuda_runtime.h>                   // cudaSetDevice

static SORTTRACKING sorttracking;
static YOLOv8* yolov8 = nullptr;
static std::vector<Object> objs;

extern const std::vector<std::vector<unsigned int>> SKELETON;
extern const std::vector<std::vector<unsigned int>> KPS_COLORS;
extern const std::vector<std::vector<unsigned int>> LIMB_COLORS;

bool trt_init(const char* lanepose_model_path,
                    char* classify_model_path,
              const char* icon_path,
              Config&     config)
{
    // 設定 GPU
    cudaSetDevice(0);

    // 建 YOLOv8
    yolov8 = new YOLOv8(lanepose_model_path);
    yolov8->make_pipe(true);

    // 初始化分類模型
    classifydetector.classify_init(classify_model_path);

    // 載入 icon 圖
    if (!IconManager::Load_Picture(icon_path)) {
        std::cerr << "\nLoad Icon Picture Failed !" << std::endl;
        return false;
    }

    return true;
}

std::vector<TrackingBox>  trt_process_frame(const cv::Mat& frame,
                       cv::Mat&       output_frame,
                       Config&        config)
{

    std::vector<TrackingBox> TrackingResult;

    if (!yolov8) {
        std::cerr << "TensorRT not initialized! Call trt_init() first.\n";
    }

    // preprocess
    yolov8->copy_from_Mat(frame, config.size);

    // invoke
    yolov8->infer();

    // proposals and draw result
    yolov8->postprocess_pose(
        objs,
        config.score_thres,
        config.iou_thres,
        config.topk,
        config.num_labels
    );

    std::vector<TrackingBox> TrackingResult = sorttracking.TrackingResult(objs);

    TrackingResult = yolov8->draw_pose(
        frame,
        output_frame,
        objs,
        SKELETON,
        KPS_COLORS,
        LIMB_COLORS,
        config.num_keypoint
    );

    return TrackingResult;
}

#else

bool trt_init(const char*,
              const char*,
              const char*,
              Config&)
{
    std::cerr << "TensorRT not enabled (USE_TENSORRT undefined).\n";
    return false;
}

void trt_process_frame(const cv::Mat&,
                       cv::Mat&,
                       Config&)
{
    // do nothing
}

#endif
