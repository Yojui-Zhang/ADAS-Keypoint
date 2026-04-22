#pragma once
#include <opencv2/core.hpp>  // 包含 cv::Scalar 和其他基本類型

// 是否啟動 CANBus 實車控車
#define CANBUS__

// 是否啟用 GPU delegate（TFLite）
// #define _GPU_delegate

// 影像/推論資料輸出開關
// #define Write_Video__
// #define Save_infer_raw_data__

// 輸入來源選擇（二擇一）
// #define _v4l2cap
#define _openCVcap


// 顯示後端選擇
// #define _opengl

// ============= 模型輸入解析度（四選一） =========
// #define _640640
// #define _640384
#define _512288
// #define _480480

// ============= 影像尺寸設定 ============

// 1920x1536 -> 1920x1080 -> 1280x720
// 1280x960 -> 1280x720 -> 1280x720

#define input_video_width 1920
#define input_video_height 1080

#define rect_video_width 1920       // 輸入裁切寬度（openCV 路徑）
#define rect_video_height 1080       // 輸入裁切高度（openCV 路徑）

#define process_video_width 1280    // 演算法處理寬度
#define process_video_height 720    // 演算法處理高度

#define output_video_width 1280     // 輸出影片寬度
#define output_video_height 720    // 輸出影片高度
#define output_video_fps 15         // 輸出影片幀率

// ============= V4L2 設定 ===================
#define V4L2_cap_num 8


// ================ 分類模型設定 ==========
#define Classify_Model_Width 60
#define Classify_Model_Height 60

#define classify_NUM_CLASS 18

// ================ 偵測模型設定 ==========
#define NUM_CLASS 7
#define PROB_THRESHOLD 0.3          // 置信度門檻
#define NMS_THRESHOLD_BBOX 0.6      // 偵測框 NMS IoU 門檻
#define NMS_THRESHOLD_LANE 0.8      // 車道線 NMS IoU 門檻
#define Keypoint_NUM 15             // 每個目標輸出 keypoint 數量

#ifdef _640640
    #define INPUT_WIDTH 640
    #define INPUT_HEIGHT 640
    #define NUM_BOXES 8400
    
#endif

#ifdef _640384
    #define INPUT_WIDTH 640
    #define INPUT_HEIGHT 384
    #define NUM_BOXES 5040
#endif

#ifdef _512288
    #define INPUT_WIDTH 512
    #define INPUT_HEIGHT 288
    #define NUM_BOXES 3456
#endif

#ifdef _480480
    #define INPUT_WIDTH 480
    #define INPUT_HEIGHT 480
    #define NUM_BOXES 4725
#endif

// ================ TensorRT Set ==========

struct Config {

    cv::Size size = cv::Size{INPUT_WIDTH, INPUT_HEIGHT};  // 模型輸入尺寸

    int num_labels = NUM_CLASS;                       // 偵測類別數
    int topk = 100;                                   // 每幀保留候選上限
    float score_thres = PROB_THRESHOLD;               // 置信度門檻
    float iou_thres = NMS_THRESHOLD_BBOX;             // NMS IoU 門檻

    // Pose
    const int num_keypoint = Keypoint_NUM;             // 關鍵點數量

    // Segmentation
    int seg_h = 160;                                 // 分割特徵圖高度
    int seg_w = 160;                                 // 分割特徵圖寬度
    int seg_channels = 32;                           // 分割特徵圖通道數

    // 偵測類別名稱（需與訓練標籤順序一致）
    const char* class_names[NUM_CLASS] = {"roadlane", "car", "rider", "person", "light", "signC", "signT"};

    // 分類類別名稱（需與分類模型標籤順序一致）
    const char* class_name_classify[classify_NUM_CLASS] = {"100km", "110km", "30km", "40km", "50km", 
                                                                            "60km", "70km", "80km", "90km", "car_left", 
                                                                            "car_normal", "car_right", "car_warning", "light_green", "light_other", 
                                                                            "light_red", "light_yellow", "sign_other"};

};


// ================ System Set ================
// OpenCV uses BGR order
const cv::Scalar RED     (  0,   0, 255);
const cv::Scalar GREEN   (  0, 255,   0);
const cv::Scalar BLUE    (255,   0,   0);
const cv::Scalar YELLOW  (  0, 255, 255);
const cv::Scalar CYAN    (255, 255,   0);
const cv::Scalar MAGENTA (255,   0, 255);
const cv::Scalar WHITE   (255, 255, 255);
const cv::Scalar BLACK   (  0,   0,   0);
const cv::Scalar GRAY    (128, 128, 128);
const cv::Scalar ORANGE  (  0, 165, 255);
const cv::Scalar PINK    (203, 192, 255);
const cv::Scalar PURPLE  (128,   0, 128);
const cv::Scalar BROWN   ( 42,  42, 165);
const cv::Scalar ARROW_COLOR(0, 255, 255); // 黃色


// ============================================


const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

const std::vector<std::vector<unsigned int>> KPS_COLORS = {{0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255}};


const std::vector<std::vector<unsigned int>> SKELETON = {{3, 4},
                                                         {5, 6},
                                                         {4, 6},
                                                         {3, 5},
                                                         {12, 13},
                                                         {14, 15},
                                                         {12, 14},
                                                         {13, 15},
                                                         {3, 7},
                                                         {4, 8},
                                                         {5, 9},
                                                         {6, 10},
                                                         {12, 7},
                                                         {13, 8},
                                                         {14, 9},
                                                         {15, 10},
                                                         {7, 8},
                                                         {9, 10},
                                                         {7, 9},
                                                         {8, 10}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0}};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
    {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49},  {72, 249, 10}, {146, 204, 23},
    {61, 219, 134}, {26, 147, 52},   {0, 212, 187},  {44, 153, 168}, {0, 194, 255},   {52, 69, 147}, {100, 115, 255},
    {0, 24, 236},   {132, 56, 255},  {82, 0, 133},   {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};


struct Object {
    int class_id;        // 偵測類別 ID
    float score;         // 偵測分數
    cv::Rect box;        // 影像座標框

    std::vector<cv::Point3f> kpts;  // x, y, visibility (TFlite)
    // std::vector<float> kps;         //                      (TensorRT)

    cv::Mat boxMask;               // 分割遮罩
};


typedef struct TrackingBox
{
    int frame;           // 幀編號
    int id;              // 追蹤 ID
    int class_id;        // 偵測類別 ID
    float score;         // 追蹤目標分數
    int classify_num;    // 分類結果 ID

    // SORT
    cv::Rect box;                                                    // 當前框（像素）
    std::vector<cv::Point3f> kpts;                                  // 當前 keypoints（Geometry 會轉世界座標）
    
    cv::Rect last_track_box;                                         // 前一幀追蹤框
    std::vector<cv::Point3f> last_track_kpts;                        // 前一幀 keypoints

    std::vector<cv::Rect> track_box_history;                        // 歷史 Bounding Boxes
    std::vector<std::vector<cv::Point3f>> track_kpt_history;        // 歷史 Keypoints (每一幀都有一組 kpts)

    // ACC
    std::vector<cv::Point3f> World_box;                             // World Box

    // Vehicle Behavior
    float target_heading_deg = std::numeric_limits<float>::quiet_NaN();
    bool  target_heading_valid = false;

} TrackingBox;
