#include "config.h"


cv::Mat rgbImg1(input_video_height, input_video_width, CV_8UC3);
cv::Mat rgbImg(output_video_height, output_video_width, CV_8UC3); 
cv::Rect rect(0, 0, output_video_width, output_video_height);


// ============================ Classify ============================


// ============================ TensorRT ============================
// ALL
int      num_labels  = 80;

int      topk        = 100;         
float    score_thres = 0.25f;
float    iou_thres   = 0.85f;

// Pose
const int num_keypoint = 17;        

// Seg
int      seg_h        = 160;        
int      seg_w        = 160;
int      seg_channels = 32;

// ============================ TensorRT ============================