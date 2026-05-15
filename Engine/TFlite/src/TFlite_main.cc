#include "TFlite_main.h"
#include "SortTracking.h"

#ifdef USE_TFLITE

#include "config.h"
#include "debug.h"
#include "../include/TFlite.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "tensorflow/lite/c/common.h"

#ifdef _GPU_delegate
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif

#include <vector>
#include <iostream>

static PoseDetector pose;
static SORTTRACKING sorttracking;

bool Classify_and_icon_init(const char* classify_model_path, const char* Icon_path)
{
    classifydetector.classify_init(classify_model_path);

    if (IconManager::Load_Picture(Icon_path) == false) {
        std::cerr << "Load Icon Picture Failed" << std::endl;
    }

    return true;
}

void tflite_set_sort_config(const SORTTRACKING::SortTrackingConfig& sort_cfg,
                            const sort_kpt::KeypointFilterConfig& kpt_cfg)
{
    sorttracking.SetConfig(sort_cfg);
    sort_kpt::ConfigureGlobalKeypointFilter(kpt_cfg);
}

bool tflite_init(const char* lanepose_model_path, const cv::Mat& first_frame)
{
    if (pose.Set_TFlite(lanepose_model_path) == false) {
        std::cerr << "[TFLite] Set_TFlite failed" << std::endl;
        return false;
    }
    if (first_frame.empty() == false) {
        pose.Calculate_Scale(first_frame, INPUT_WIDTH, INPUT_HEIGHT);
    }
    return true;
}

std::vector<TrackingBox> tflite_run_frame(const cv::Mat& frame,
                                          cv::Mat& out_bgr,
                                          int classify_model_width,
                                          int classify_model_height,
                                          bool draw_visuals,
                                          std::vector<Object>* raw_objects)
{
    pose.get_input_data_fp32(frame,
                             pose.input_data,
                             INPUT_HEIGHT, INPUT_WIDTH,
                             pose.mean, pose.scale,
                             pose.new_width, pose.new_height,
                             pose.top, pose.bottom, pose.left, pose.right);

    if (pose.interpreter->Invoke() == kTfLiteOk) {
    } else {
        std::cerr << "[TFLite] Invoke failed" << std::endl;
    }

#ifdef Save_infer_raw_data__
    (void)SaveOutputTensorToTxt(pose.interpreter.get(),
                                0,
                                "yolov8_output.txt");
#endif

    std::vector<Object> objs;
    std::vector<TrackingBox> TrackingResult;

    pose.generate_proposals(pose.yolov8_output,
                            PROB_THRESHOLD,
                            objs,
                            pose.scale_factor,
                            pose.top, pose.left);

    pose.nms(objs, NMS_THRESHOLD_BBOX, NMS_THRESHOLD_LANE);

    if (raw_objects != nullptr) {
        *raw_objects = objs;
    }

    TrackingResult = sorttracking.TrackingResult(objs);

    if (draw_visuals) {
        out_bgr = frame;
        pose.draw_objects(frame,
                          TrackingResult,
                          out_bgr,
                          classify_model_width,
                          classify_model_height);
    } else {
        out_bgr = frame.clone();
    }
    return TrackingResult;
}

#else

bool Classify_and_icon_init(const char*, const char*) { return false; }
void tflite_set_sort_config(const SORTTRACKING::SortTrackingConfig&, const sort_kpt::KeypointFilterConfig&) {}
bool tflite_init(const char*, const cv::Mat&) { return false; }
std::vector<TrackingBox> tflite_run_frame(const cv::Mat&, cv::Mat&, int, int, bool, std::vector<Object>*) { return {}; }

#endif
