#pragma once

#include "config.h"
#include <opencv2/core/core.hpp>

bool trt_init(const char* lanepose_model_path,
                    char* classify_model_path,
              const char* icon_path,
              Config&     config);

void trt_process_frame(const cv::Mat& frame,
                       cv::Mat&       output_frame,
                       Config&        config);
