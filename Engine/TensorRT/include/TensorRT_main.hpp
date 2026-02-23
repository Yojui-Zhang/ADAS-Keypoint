#pragma once

#include "config.h"
#include "SortTracking.h"
#include "KeypointFilterSwitch.h"
#include <opencv2/core/core.hpp>

void trt_set_sort_config(const SORTTRACKING::SortTrackingConfig& sort_cfg,
                         const sort_kpt::KeypointFilterConfig& kpt_cfg);

bool trt_init(const char* lanepose_model_path,
                    char* classify_model_path,
              const char* icon_path,
              Config&     config);

std::vector<TrackingBox>  trt_process_frame(const cv::Mat&       frame,
                                                  cv::Mat&       output_frame,
                                                  Config&        config);
