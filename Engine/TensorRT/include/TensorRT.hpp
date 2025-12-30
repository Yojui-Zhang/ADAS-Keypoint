#ifndef DETECT_NORMAL_YOLOV8_HPP
#define DETECT_NORMAL_YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>  // OpenCL 支援

#include "SortTracking.h"
#include "draw_icon.h"


using namespace det;
using namespace cv::dnn;

class classifyDetector{
public:
    void classify_init(char* classify_model_path);
    cv::Mat cropObjects(const Mat& frame, const TrackingBox &obj, int classify_model_width, int classify_model_height);
};

class YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path);
    ~YOLOv8();

    void                 make_pipe(bool warmup = true);
    void                 copy_from_Mat(const cv::Mat& image);
    void                 copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void                 letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void                 infer();

// =====================================================================================================
    void postprocess_classify(std::vector<Object>& objs);

    static void draw_classify(const cv::Mat&                  image,
                             cv::Mat&                        res,
                             const std::vector<Object>&      objs);
// =====================================================================================================
    void                 postprocess_detect(std::vector<Object>& objs,
                                     float                score_thres = 0.25f,
                                     float                iou_thres   = 0.65f,
                                     int                  topk        = 100,
                                     int                  num_labels  = 80);
    static void          draw_objects(const cv::Mat&                                image,
                                      cv::Mat&                                      res,
                                      const std::vector<Object>&                    objs,
                                      const std::vector<std::vector<unsigned int>>& COLORS);
// =====================================================================================================
    void postprocess_pose(std::vector<Object>& objs, float score_thres = 0.25f, float iou_thres = 0.65f, int topk = 100, int num_labels  = 80);
 
    static void          draw_pose(const cv::Mat&                                image,
                                                     cv::Mat&                                res,
                                               const std::vector<Object>&                    objs,
                                               const std::vector<std::vector<unsigned int>>& SKELETON,
                                               const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                                               const std::vector<std::vector<unsigned int>>& LIMB_COLORS,
                                               const int num_keypoint);
// =====================================================================================================
    void                 postprocess_seg(std::vector<Object>& objs,
                                     float                score_thres  = 0.25f,
                                     float                iou_thres    = 0.65f,
                                     int                  topk         = 100,
                                     int                  seg_channels = 32,
                                     int                  seg_h        = 160,
                                     int                  seg_w        = 160);
    static void          draw_seg(const cv::Mat&                                image,
                                      cv::Mat&                                      res,
                                      const std::vector<Object>&                    objs,
                                      const std::vector<std::vector<unsigned int>>& COLORS,
                                      const std::vector<std::vector<unsigned int>>& MASK_COLORS);
// =====================================================================================================
    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    PreParam pparam;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};
// ========================================================

Net net;
classifyDetector classifydetector;
Config config;

void classifyDetector::classify_init(char* classify_model_path)
{
#ifdef _GPU_delegate
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        cout << "OpenCL is enabled!" << endl;
    } else {
        cout << "OpenCL is not supported on this device." << endl;
    }

#endif
    net = readNetFromONNX(classify_model_path);

    if (net.empty()) {
        cerr << "Failed to load ONNX model!" << endl;
    }
#ifdef _GPU_delegate
    // 使用 OpenCL 進行推論加速
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);  // 讓 OpenCV 自動選擇最佳後端
    net.setPreferableTarget(DNN_TARGET_OPENCL);     // 指定使用 OpenCL 進行加速
#endif
}

// 裁剪偵測到的物件
cv::Mat classifyDetector::cropObjects(const Mat& frame, const TrackingBox &obj, int classify_model_width, int classify_model_height ) {

    Mat crop_image;

    int width = classify_model_width;
    int height = classify_model_height;

    // 取得偵測物件的邊界框
    cv::Rect roi = obj.box;

    // 確保裁剪區域不超過影像邊界
    roi &= Rect(0, 0, frame.cols, frame.rows);

    // 裁剪影像
    Mat croppedObject = frame(roi).clone(); // 需要 clone() 避免引用原圖

    resize(croppedObject, crop_image, cv::Size(width, height));

    return crop_image;
}

// ========================================================

YOLOv8::YOLOv8(const std::string& engine_file_path)
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);

#ifdef TRT_10
    this->num_bindings = this->engine->getNbIOTensors();
#else
    this->num_bindings = this->num_bindings = this->engine->getNbBindings();
#endif

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding        binding;
        nvinfer1::Dims dims;

#ifdef TRT_10
        std::string        name  = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
#else
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
#endif
        binding.name  = name;
        binding.dsize = type_to_size(dtype);
#ifdef TRT_10
        bool IsInput = engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
#else
        bool IsInput = engine->bindingIsInput(i);
#endif
        if (IsInput) {
            this->num_inputs += 1;
#ifdef TRT_10
            dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->context->setInputShape(name.c_str(), dims);
#else
            dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
        }
        else {
#ifdef TRT_10
            dims = this->context->getTensorShape(name.c_str());
#else
            dims = this->context->getBindingDimensions(i);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8::~YOLOv8()
{
#ifdef TRT_10
    delete this->context;
    delete this->engine;
    delete this->runtime;
#else
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
#endif
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}
void YOLOv8::make_pipe(bool warmup)
{

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->context->setInputShape(name, bindings.dims);
        this->context->setTensorAddress(name, d_ptr);
#endif
    }

    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->context->setTensorAddress(name, d_ptr);
#endif
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    out.create({1, 3, (int)inp_h, (int)inp_w}, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float*)out.data);
    cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w);
    cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w * 2);

    channels[0].convertTo(c2, CV_32F, 1 / 255.f);
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);

    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}

void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    int      width      = in_binding.dims.d[3];
    int      height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});
#endif
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
#endif
}

void YOLOv8::infer()
{
#ifdef TRT_10
    this->context->enqueueV3(this->stream);
#else
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
#endif
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

// ============================================ classify ============================================

void YOLOv8::postprocess_classify(std::vector<Object>& objs)
{
    objs.clear();
    auto num_cls = this->output_bindings[0].dims.d[1];

    float* max_ptr =
        std::max_element(static_cast<float*>(this->host_ptrs[0]), static_cast<float*>(this->host_ptrs[0]) + num_cls);
    Object obj;
    obj.class_id = std::distance(static_cast<float*>(this->host_ptrs[0]), max_ptr);
    obj.score  = *max_ptr;
    objs.push_back(obj);
}

void YOLOv8::draw_classify(const cv::Mat&                  image,
                              cv::Mat&                        res,
                              const std::vector<Object>&      objs)
{
    res = image.clone();
    char   text[256];
    Object obj = objs[0];
    sprintf(text, "%s %.1f%%", config.class_names[obj.class_id], obj.score * 100);

    int      baseLine   = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
    int      x          = 10;
    int      y          = 10;

    if (y > res.rows)
        y = res.rows;

    cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
    cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
}



// ============================================ detect ============================================

void YOLOv8::postprocess_detect(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int num_labels)
{
    objs.clear();
    int num_channels = this->output_bindings[0].dims.d[1];
    int num_anchors  = this->output_bindings[0].dims.d[2];

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::Rect> bboxes;
    std::vector<float>    scores;
    std::vector<int>      labels;
    std::vector<int>      indices;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto  row_ptr    = output.row(i).ptr<float>();
        auto  bboxes_ptr = row_ptr;
        auto  scores_ptr = row_ptr + 4;
        auto  max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score      = *max_s_ptr;
        if (score > score_thres) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            int              label = max_s_ptr - scores_ptr;
            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.box  = bboxes[i];
        obj.score  = scores[i];
        obj.class_id = labels[i];
        objs.push_back(obj);
        cnt += 1;
    }
}


void YOLOv8::draw_objects(const cv::Mat&                                image,
                          cv::Mat&                                      res,
                          const std::vector<Object>&                    objs,
                          const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    for (auto& obj : objs) {
        cv::Scalar color = cv::Scalar(COLORS[obj.class_id][0], COLORS[obj.class_id][1], COLORS[obj.class_id][2]);
        cv::rectangle(res, obj.box, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", config.class_names[obj.class_id], obj.score * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.box.x;
        int y = (int)obj.box.y + 1;

        if (y > res.rows) {
            y = res.rows;
        }
        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}

// ============================================ pose ============================================

void YOLOv8::postprocess_pose(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int num_labels)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::Rect>           bboxes;
    std::vector<float>              scores;
    std::vector<int>                labels;
    std::vector<int>                indices;
    std::vector<std::vector<cv::Point3f>> kpss;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr    = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + 4;
        auto kps_ptr    = row_ptr + 5;
        auto max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score      = *max_s_ptr;

        if (score > score_thres) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            int   label = max_s_ptr - scores_ptr;

            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;
            std::vector<cv::Point3f> kps;
            for (int k = 0; k < 17; k++) {
                float kps_x = (*(kps_ptr + 3 * k) - dw) * ratio;
                float kps_y = (*(kps_ptr + 3 * k + 1) - dh) * ratio;
                float kps_s = *(kps_ptr + 3 * k + 2);

                kps_x = clamp(kps_x, 0.f, width);
                kps_y = clamp(kps_y, 0.f, height);

                // x, y, score -> 用 Point3f 的 (x, y, z)
                kps.emplace_back(kps_x, kps_y, kps_s);
            }

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            kpss.push_back(kps);
        }
    }

#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.box  = bboxes[i];
        obj.score  = scores[i];
        obj.class_id = labels[i];
        obj.kpts   = kpss[i];
        objs.push_back(obj);
        cnt += 1;
    }
}


void                     YOLOv8::draw_pose( const cv::Mat&                                image,
                                                  cv::Mat&                                res,
                                            const std::vector<TrackingBox>&               objs,
                                            const std::vector<std::vector<unsigned int>>& SKELETON,
                                            const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                                            const std::vector<std::vector<unsigned int>>& LIMB_COLORS,
                                            const int num_keypoint)
{

    int icon_light_num = 3;
    int icon_sign_num = 9;

    res                 = image.clone();
    const int num_point = num_keypoint;


    for (auto& obj : objs) {

        bool classify_light__ = false;
        int traffic_class_num;

        if(obj.class_id == 1 && obj.box.x >= 400 && obj.box.x <= 880 && obj.box.y >= 250){

            Mat crop_image;
            crop_image = classifydetector.cropObjects(image, obj, Classify_Model_Width, Classify_Model_Height);  //to crop_image

            // 調整輸入大小 (根據 ONNX 模型需求)
            Mat blob;
            Size inputSize(Classify_Model_Width, Classify_Model_Height);  // 根據模型需求調整
            blobFromImage(crop_image, blob, 1.0 / 255, inputSize, Scalar(), true, false);

            // 設定模型輸入
            net.setInput(blob);

            Mat classify_output = net.forward();

            // 解析結果
            Point classId;
            double confidence;
            minMaxLoc(classify_output, nullptr, &confidence, nullptr, &classId);

            // cout << "Predicted Class: " << classifydetector.class_name_classify[classId.x] << ", Confidence: " << confidence << endl;

            classify_light__ = true;
            traffic_class_num = classId.x;

        }
        else if( ( obj.class_id == 4 || obj.class_id == 5 || obj.class_id == 6) && obj.box.y <= 250 ){

            Mat crop_image;
            crop_image = classifydetector.cropObjects(image, obj, Classify_Model_Width, Classify_Model_Height);  //to crop_image

            // 調整輸入大小 (根據 ONNX 模型需求)
            Mat blob;
            Size inputSize(Classify_Model_Width, Classify_Model_Height);  // 根據模型需求調整
            blobFromImage(crop_image, blob, 1.0 / 255, inputSize, Scalar(), true, false);

            // 設定模型輸入
            net.setInput(blob);

            Mat classify_output = net.forward();

            // 解析結果
            Point classId;
            double confidence;
            minMaxLoc(classify_output, nullptr, &confidence, nullptr, &classId);

            // cout << "Predicted Class: " << classifydetector.class_name_classify[classId.x] << ", Confidence: " << confidence << endl;

            classify_light__ = true;
            traffic_class_num = classId.x;

            if((traffic_class_num == 13 || traffic_class_num == 15 || traffic_class_num == 16)){
                icon_light_num = traffic_class_num;
            }
            if(traffic_class_num >= 0 && traffic_class_num <= 8 ){
                icon_sign_num = traffic_class_num;
            }

        }
        
        if(obj.class_id < 2){

            auto& kps = obj.kpts;  // std::vector<cv::Point3f>
            for (int k = 0; k < num_point + 2; k++) {
                if (k < num_point) {
                    const cv::Point3f& pt = kps[k];
                    int   kps_x = std::round(pt.x);
                    int   kps_y = std::round(pt.y);
                    float kps_s = pt.z;          // score

                    if (kps_s > 0.5f && obj.class_id == 0) {
                        cv::Scalar kps_color = cv::Scalar(0, 255, 0);
                        cv::circle(res, {kps_x, kps_y}, 4, kps_color, -1);
                    }
                }

                const auto& ske = SKELETON[k];
                const cv::Point3f& p1 = kps[ske[0] - 1];
                const cv::Point3f& p2 = kps[ske[1] - 1];

                int   pos1_x = std::round(p1.x);
                int   pos1_y = std::round(p1.y);
                int   pos2_x = std::round(p2.x);
                int   pos2_y = std::round(p2.y);
                float pos1_s = p1.z;
                float pos2_s = p2.z;

                if (pos1_s > 0.5f && pos2_s > 0.5f && obj.class_id == 1) {
                    cv::Scalar limb_color = cv::Scalar(255, 0, 0);
                    cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
                }
            }
        }

        if(obj.class_id >= 2){
            cv::rectangle(res, obj.box, {255, 0, 0}, 2);
        }

        if(obj.class_id != 0){
            // Draw class label
            char text[256];
            if(classify_light__ == false){
                sprintf(text, "%s %.1f%%", config.class_names[obj.class_id], obj.score * 100);
            }
            else if(classify_light__ == true){
                sprintf(text, "%s %.1f%%", config.class_name_classify[traffic_class_num], obj.score * 100);
            }

            int      baseLine   = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

            int x = (int)obj.box.x;
            int y = (int)obj.box.y + 1;

            if (y > res.rows)
                y = res.rows;

            // cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {255, 255, 255}, -1);
            cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {0, 0, 0}, 3);
            cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
        }
    }
    // draw icon light
    res = IconManager::Draw_Icon_Light(res, icon_light_num);

    // draw icon sign
    res = IconManager::Draw_Icon_Sign(res, icon_sign_num);

    // return TrackingResult;
}

// ============================================ seg ============================================

void YOLOv8::postprocess_seg(
    std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int seg_channels, int seg_h, int seg_w)
{
    objs.clear();
    auto input_h = this->input_bindings[0].dims.d[2];
    auto input_w = this->input_bindings[0].dims.d[3];
    int  num_channels, num_anchors, num_classes;
    bool flag = false;
    int  bid;
    int  bcnt = -1;
    for (auto& o : this->output_bindings) {
        bcnt += 1;
        if (o.dims.nbDims == 3) {
            num_channels = o.dims.d[1];
            num_anchors  = o.dims.d[2];
            flag         = true;
            bid          = bcnt;
        }
    }
    assert(flag);
    num_classes = num_channels - seg_channels - 4;

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[bid]));
    output         = output.t();

    cv::Mat protos = cv::Mat(seg_channels, seg_h * seg_w, CV_32F, static_cast<float*>(this->host_ptrs[1 - bid]));

    std::vector<int>      labels;
    std::vector<float>    scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat>  mask_confs;
    std::vector<int>      indices;

    for (int i = 0; i < num_anchors; i++) {
        auto  row_ptr        = output.row(i).ptr<float>();
        auto  bboxes_ptr     = row_ptr;
        auto  scores_ptr     = row_ptr + 4;
        auto  mask_confs_ptr = row_ptr + 4 + num_classes;
        auto  max_s_ptr      = std::max_element(scores_ptr, scores_ptr + num_classes);
        float score          = *max_s_ptr;
        if (score > score_thres) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            int              label = max_s_ptr - scores_ptr;
            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;

            cv::Mat mask_conf = cv::Mat(1, seg_channels, CV_32F, mask_confs_ptr);

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            mask_confs.push_back(mask_conf);
        }
    }

#if defined(BATCHED_NMS)
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    cv::Mat masks;
    int     cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object   obj;
        obj.class_id = labels[i];
        obj.box  = tmp;
        obj.score  = scores[i];
        masks.push_back(mask_confs[i]);
        objs.push_back(obj);
        cnt += 1;
    }

    if (masks.empty()) {
        // masks is empty
    }
    else {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat   = matmulRes.reshape(indices.size(), {seg_h, seg_w});

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = dw / input_w * seg_w;
        int scale_dh = dh / input_h * seg_h;

        cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

        for (int i = 0; i < indices.size(); i++) {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].box) > 0.5f;
        }
    }
}

void YOLOv8::draw_seg(const cv::Mat&                                    image,
                            cv::Mat&                                      res,
                            const std::vector<Object>&                    objs,
                            const std::vector<std::vector<unsigned int>>& COLORS,
                            const std::vector<std::vector<unsigned int>>& MASK_COLORS)
{
    res          = image.clone();
    cv::Mat mask = image.clone();
    for (auto& obj : objs) {
        int        idx   = obj.class_id;
        cv::Scalar color = cv::Scalar(COLORS[idx][0], COLORS[idx][1], COLORS[idx][2]);
        cv::Scalar mask_color =
            cv::Scalar(MASK_COLORS[idx % 20][0], MASK_COLORS[idx % 20][1], MASK_COLORS[idx % 20][2]);
        cv::rectangle(res, obj.box, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", config.class_names[idx], obj.score * 100);
        mask(obj.box).setTo(mask_color, obj.boxMask);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.box.x;
        int y = (int)obj.box.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
    cv::addWeighted(res, 0.5, mask, 0.8, 1, res);
}


#endif  // DETECT_NORMAL_YOLOV8_HPP
