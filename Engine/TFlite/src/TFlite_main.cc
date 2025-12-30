#include "TFlite_main.h"
#include "SortTracking.h"

#ifdef USE_TFLITE

#include "config.h"
#include "debug.h"
#include "../include/TFlite.h"

// TensorFlow Lite
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

// ======================= Classify and icon init =======================
bool Classify_and_icon_init(const char* classify_model_path, const char* Icon_path){

    classifydetector.classify_init(classify_model_path);

    if ( IconManager::Load_Picture(Icon_path) != true){
        std::cerr << "\nLoad Icon Picture Failed !" << std:: endl;
    }

    return true;

}
// ======================= TFlite Process =======================

bool tflite_init(const char* lanepose_model_path, const cv::Mat& first_frame)
{
    if (!pose.Set_TFlite(lanepose_model_path)) {
        std::cerr << "[TFLite] Set_TFlite failed\n";
        return false;
    }
    if (!first_frame.empty()) {
        pose.Calculate_Scale(first_frame, INPUT_WIDTH, INPUT_HEIGHT);
    }
    return true;
}

std::vector<TrackingBox> tflite_run_frame(const cv::Mat& frame,
                                                cv::Mat& out_bgr,
                                          int classify_model_width,
                                          int classify_model_height)
{

    // preprocess
    pose.get_input_data_fp32(frame,
                             pose.input_data,
                             INPUT_HEIGHT, INPUT_WIDTH,
                             pose.mean, pose.scale,
                             pose.new_width, pose.new_height,
                             pose.top, pose.bottom, pose.left, pose.right);

    // invoke
    if (pose.interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "[TFLite] Invoke failed!\n";
    }

    // dump output
#ifdef Save_infer_raw_data__
    (void)SaveOutputTensorToTxt(pose.interpreter.get(), /*output_index=*/0,
                                "yolov8_output.txt");
#endif

    // proposals and draw result
    std::vector<Object> objs;
    std::vector<TrackingBox> TrackingResult;

    pose.generate_proposals(pose.yolov8_output,
                            PROB_THRESHOLD,
                            objs,
                            pose.scale_factor,
                            pose.top, pose.left);

    pose.nms(objs, NMS_THRESHOLD_BBOX, NMS_THRESHOLD_LANE);

    TrackingResult = sorttracking.TrackingResult(objs);

    pose.draw_objects(frame, TrackingResult, out_bgr,
                                classify_model_width, classify_model_height);
    return TrackingResult;
}

#else

bool Classify_and_icon_init(const char* classify_model_path, const char* Icon_path) {return false; }
bool tflite_init(const char*, const cv::Mat&) { return false; }
bool tflite_run_frame(const cv::Mat&, cv::Mat&, int, int) { return false; }

#endif
