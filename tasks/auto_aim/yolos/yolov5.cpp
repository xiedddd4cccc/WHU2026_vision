#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include <opencv2/dnn.hpp>
#include "tasks/auto_aim/yolos/trt_utils.hpp" 

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace auto_aim
{

// (构造函数, 析构函数 ... 保持不变)
YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  // ... (保留你原来的整个构造函数代码) ...
  try {
    auto yaml = YAML::LoadFile(config_path);
    model_path_ = yaml["yolov5_model_path"].as<std::string>();
    if (yaml["threshold"]) binary_threshold_ = yaml["threshold"].as<double>();
    if (yaml["min_confidence"]) min_confidence_ = yaml["min_confidence"].as<double>();
    if (yaml["score_threshold"]) score_threshold_ = yaml["score_threshold"].as<double>();
    if (yaml["nms_threshold"]) nms_threshold_ = yaml["nms_threshold"].as<double>();
    int x = 0, y = 0, width = -1, height = -1;
    if (yaml["roi"]) {
      x = yaml["roi"]["x"].as<int>();
      y = yaml["roi"]["y"].as<int>();
      width = yaml["roi"]["width"].as<int>();
      height = yaml["roi"]["height"].as<int>();
      use_roi_ = yaml["use_roi"].as<bool>();
    }
    if (yaml["use_traditional"]) use_traditional_ = yaml["use_traditional"].as<bool>();
    roi_ = cv::Rect(x, y, width, height);
    offset_ = cv::Point2f(x, y);
    save_path_ = "imgs";
    std::filesystem::create_directory(save_path_);
    std::ifstream engineFile(model_path_, std::ios::binary);
    if (!engineFile.is_open()) {
      tools::logger()->error("YOLOV5: Failed to open engine file: {}", model_path_);
      return;
    }
    engineFile.seekg(0, std::ios::end);
    std::streamsize fsize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(static_cast<size_t>(fsize));
    if (!engineFile.read(engineData.data(), fsize)) {
      tools::logger()->error("YOLOV5: Failed to read engine file: {}", model_path_);
      return;
    }
    engineFile.close();
    nvinfer1::IRuntime* runtime_ptr = nvinfer1::createInferRuntime(mLogger_);
    if (!runtime_ptr) {
      tools::logger()->error("YOLOV5: createInferRuntime failed");
      return;
    }
    mRuntime.reset(runtime_ptr);
    nvinfer1::ICudaEngine* engine_ptr = mRuntime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine_ptr) {
      tools::logger()->error("YOLOV5: deserializeCudaEngine failed");
      return;
    }
    mEngine.reset(engine_ptr);
    nvinfer1::IExecutionContext* ctx_ptr = mEngine->createExecutionContext();
    if (!ctx_ptr) {
      tools::logger()->error("YOLOV5: createExecutionContext failed");
      return;
    }
    mContext.reset(ctx_ptr);
    CHECK_CUDA(cudaStreamCreate(&mStream));
    allocateBindings();
    tools::logger()->info("YOLOV5: TensorRT engine loaded.");
  }
  catch (const std::exception & e) {
    tools::logger()->error("YOLOV5: Exception in ctor: {}", e.what());
  }
}
YOLOV5::~YOLOV5()
{
  try {
    freeBindings();
    if (mStream) {
      cudaError_t err = cudaStreamDestroy(mStream);
      if (err != cudaSuccess) {
        tools::logger()->warn("YOLOV5: cudaStreamDestroy failed: {}", cudaGetErrorString(err));
      }
      mStream = nullptr;
    }
    tools::logger()->info("YOLOV5: resources released.");
  } catch (...) {}
}


// --------------- allocate / free bindings ---------------
void YOLOV5::allocateBindings()
{
  if (!mEngine) return;

  int nbIOTensors = mEngine->getNbIOTensors();
  if (nbIOTensors <= 0) {
    tools::logger()->error("YOLOV5: engine has no I/O tensors");
    return;
  }
  for (int i = 0; i < nbIOTensors; ++i) {
    const char* tname = mEngine->getIOTensorName(i);
    // ... (省略 Tname, mode, dtype, dims 的获取) ...
    std::string name(tname);
    nvinfer1::DataType dtype = mEngine->getTensorDataType(name.c_str());
    nvinfer1::TensorIOMode mode = mEngine->getTensorIOMode(name.c_str());
    nvinfer1::Dims dims = mEngine->getTensorShape(name.c_str());

    size_t elems = getTensorVolume(dims);
    size_t bytes = elems * getElementSize(dtype);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      mInputTensorName_ = name;
      mInputDims_ = dims;
      mInputBytes_ = bytes;
      mInputDtype_ = dtype; 

      cudaError_t err = cudaMalloc(&mGpuBuffers_[0], bytes);
      if (err != cudaSuccess) {
        tools::logger()->error("YOLOV5: cudaMalloc input failed: {}", cudaGetErrorString(err));
        mGpuBuffers_[0] = nullptr;
      } else {
        tools::logger()->info("YOLOV5: Alloc TRT input '{}' elems={} bytes={} ({})",
                              name, elems, bytes, (dtype == nvinfer1::DataType::kHALF ? "FP16" : "FP32"));
        // -----------------------------------------------------------------
        // ⬇️ ⬇️ [关键修改] 预分配 "固定内存" ⬇️ ⬇️
        // -----------------------------------------------------------------
        // 我们需要一个 FP32 大小的固定内存，因为 blobFromImage 输出的是 FP32
        // elems = N*C*H*W
        size_t fp32_bytes = elems * sizeof(float);
        
        // 按照 (1, N*C*H*W) 的扁平形状分配
        mCpuPinnedBlob_.create(1, fp32_bytes / sizeof(float), CV_32F);
        
        tools::logger()->info("YOLOV5: Pre-allocated Pinned CPU buffer ({} bytes)", fp32_bytes);
        // -----------------------------------------------------------------
      }
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      mOutputTensorName_ = name;
      mOutputDims_ = dims;
      mOutputBytes_ = bytes;
      mCpuOutputBuffer_.resize(elems);
      cudaError_t err = cudaMalloc(&mGpuBuffers_[1], bytes);
      if (err != cudaSuccess) {
        tools::logger()->error("YOLOV5: cudaMalloc output failed: {}", cudaGetErrorString(err));
        mGpuBuffers_[1] = nullptr;
      } else {
         tools::logger()->info("YOLOV5: Alloc TRT output '{}' elems={} bytes={} ({})",
                              name, elems, bytes, (dtype == nvinfer1::DataType::kHALF ? "FP16" : "FP32"));
      }
    } else {
      tools::logger()->warn("YOLOV5: unknown TensorIOMode for '{}'", name);
    }
  }
}

void YOLOV5::freeBindings()
{
  for (int i = 0; i < 2; ++i) {
    if (mGpuBuffers_[i]) {
      cudaError_t err = cudaFree(mGpuBuffers_[i]);
      if (err != cudaSuccess) {
        tools::logger()->warn("YOLOV5: cudaFree buffer[{}] failed: {}", i, cudaGetErrorString(err));
      }
      mGpuBuffers_[i] = nullptr;
    }
  }
  mCpuOutputBuffer_.clear();
  mInputBytes_ = 0;
  mOutputBytes_ = 0;
  mInputDtype_ = nvinfer1::DataType::kFLOAT;

  mGpuBlobFp32_.release();
  mGpuBlobFp16_.release();

  // -----------------------------------------------------------------
  // ⬇️ ⬇️ [关键修改] 释放 "固定内存" ⬇️ ⬇️
  // -----------------------------------------------------------------
  mCpuPinnedBlob_.release();
}

// ----------------- detect / preprocess / infer / postprocess -----------------
std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("YOLOV5::detect empty image");
    return {};
  }
  cv::Mat bgr_img = use_roi_ ? raw_img(roi_) : raw_img;
  if (mInputDims_.nbDims < 4) {
    tools::logger()->error("YOLOV5: invalid input dims");
    return {};
  }
  int inN = mInputDims_.d[0] > 0 ? mInputDims_.d[0] : 1;
  int inC = mInputDims_.d[1];
  int inH = mInputDims_.d[2];
  int inW = mInputDims_.d[3];

  // -----------------------------------------------------------------
  // ⬇️ ⬇️ [关键修改] 终极高性能流水线 ⬇️ ⬇️
  // -----------------------------------------------------------------

  // 1. [CPU] resize (快)
  cv::Mat resized;
  cv::resize(bgr_img, resized, cv::Size(inW, inH));
  
  // 2. [CPU] 将 blobFromImage **直接写入固定内存**
  
  // 2a. 从我们的 HostMem (mCpuPinnedBlob_) 获取一个 cv::Mat "头" (视图)
  // mCpuPinnedBlob_ 是 [1, N*C*H*W]
  cv::Mat blob_pinned_header = mCpuPinnedBlob_.createMatHeader();
  
  // 2b. 将这个视图 reshape 为 4D [N, C, H, W]，以便 blobFromImage 接受它
  std::vector<int> blob_shape = {inN, inC, inH, inW};
  cv::Mat blob_for_dnn = blob_pinned_header.reshape(1, blob_shape); // 0 耗时

  // 2c. [CPU] OpenCV 将数据**直接写入固定内存**，而不是标准内存
  cv::dnn::blobFromImage(resized, blob_for_dnn, 1.0 / 255.0, cv::Size(inW, inH), cv::Scalar(), true, false, CV_32F);
  // (blob_for_dnn 现在是 CPU 上的 FP32, 但位于 Pinned Memory 中)

  // 3. [H2D] 将 FP32 blob 从 "固定内存" **真正异步** 上传到 GPU
  auto cv_stream = cv::cuda::StreamAccessor::wrapStream(mStream);
  
  // 我们上传 blob_for_dnn, 它指向 mCpuPinnedBlob_
  mGpuBlobFp32_.upload(blob_for_dnn, cv_stream); // 0 CPU 耗时

  // 4. [GPU] 在 GPU 上执行 FP32 -> FP16 转换 (0 耗时)
  mGpuBlobFp32_.convertTo(mGpuBlobFp16_, CV_16S, cv_stream);

  if (!mGpuBuffers_[0]) {
    tools::logger()->error("YOLOV5: input GPU buffer not allocated");
    return {};
  }
  
  // 5. [D2D] 将 GpuMat (FP16) 拷贝到 TRT 缓冲区 (0 耗时)
  CHECK_CUDA(cudaMemcpyAsync(mGpuBuffers_[0], mGpuBlobFp16_.data, mInputBytes_, cudaMemcpyDeviceToDevice, mStream));

  // -----------------------------------------------------------------
  // ⬆️ ⬆️ [修改完毕] ⬆️ ⬆️
  // -----------------------------------------------------------------

  // 6. [GPU] 执行推理
  mContext->setTensorAddress(mInputTensorName_.c_str(), mGpuBuffers_[0]);
  mContext->setTensorAddress(mOutputTensorName_.c_str(), mGpuBuffers_[1]);
  if (!mContext->enqueueV3(mStream)) {
    tools::logger()->error("YOLOV5: enqueueV3 failed");
    return {};
  }

  // 7. [D2H] 异步拷回结果
  if (!mGpuBuffers_[1]) {
    tools::logger()->error("YOLOV5: output GPU buffer not allocated");
    return {};
  }
  size_t cpu_output_bytes = mCpuOutputBuffer_.size() * sizeof(float);
  CHECK_CUDA(cudaMemcpyAsync(mCpuOutputBuffer_.data(), mGpuBuffers_[1], cpu_output_bytes, cudaMemcpyDeviceToHost, mStream));
  
  // 8. [SYNC] CPU 在这里等待所有 GPU 工作 (1-7) 完成
  CHECK_CUDA(cudaStreamSynchronize(mStream));

  // 9. [CPU] 后处理
  // ... (build output Mat, 与你修复后的代码相同) ...
  int out_rows = 1;
  int out_cols = static_cast<int>(mCpuOutputBuffer_.size());
  if (mOutputDims_.nbDims >= 3) {
      out_rows = mOutputDims_.d[mOutputDims_.nbDims - 2];
      out_cols = mOutputDims_.d[mOutputDims_.nbDims - 1];
  } else if (mOutputDims_.nbDims == 2) {
      out_rows = mOutputDims_.d[0];
      out_cols = mOutputDims_.d[1];
  }
  if (static_cast<size_t>(out_rows * out_cols) != mCpuOutputBuffer_.size() && mCpuOutputBuffer_.size() > 0) {
    if (mOutputDims_.nbDims >= 2 && mOutputDims_.d[mOutputDims_.nbDims - 1] > 0) {
        out_cols = mOutputDims_.d[mOutputDims_.nbDims - 1];
        out_rows = mCpuOutputBuffer_.size() / out_cols;
    }
    if (static_cast<size_t>(out_rows * out_cols) != mCpuOutputBuffer_.size()) {
       out_rows = 1;
       out_cols = static_cast<int>(mCpuOutputBuffer_.size());
    }
  }
  cv::Mat output(out_rows, out_cols, CV_32F, mCpuOutputBuffer_.data());
  
  return parse(1.0, output, bgr_img, frame_count);
}

// ----------------- parse / postprocess / helpers -----------------
// (这部分函数完全不变，保留你所有的坐标修复逻辑)
std::list<Armor> YOLOV5::parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  double orig_h = (double)bgr_img.rows;
  double orig_w = (double)bgr_img.cols;
  double resized_h = (mInputDims_.nbDims >= 3) ? (double)mInputDims_.d[2] : orig_h;
  double resized_w = (mInputDims_.nbDims >= 4) ? (double)mInputDims_.d[3] : orig_w;
  if (resized_h == 0) resized_h = orig_h;
  if (resized_w == 0) resized_w = orig_w;
  double scale_w = orig_w / resized_w;
  double scale_h = orig_h / resized_h;

  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  if (output.empty() || output.type() != CV_32F) {
      tools::logger()->error("YOLOV5::parse: Invalid output mat.");
      return {};
  }
  
  for (int r = 0; r < output.rows; ++r) {
    if (output.cols < 22) {
        if(r == 0) tools::logger()->warn("YOLOV5::parse: Output cols ({}) < 22. May truncate.", output.cols);
    }
        
    double score = sigmoid(output.at<float>(r, 8));
    if (score < score_threshold_) continue;

    cv::Mat color_scores = output.row(r).colRange(9, std::min(13, output.cols));
    cv::Mat class_scores = output.row(r).colRange(13, std::min(22, output.cols));
    cv::Point class_id, color_id;
    double score_class = 0.0, score_color = 0.0;
    if (class_scores.cols > 0) cv::minMaxLoc(class_scores, NULL, &score_class, NULL, &class_id);
    if (color_scores.cols > 0) cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);

    std::vector<cv::Point2f> armor_key_points;
    auto val_at = [&](int rr, int cc)->float {
      return (cc < output.cols) ? output.at<float>(rr, cc) : 0.f;
    };
    
    armor_key_points.push_back({ val_at(r,0) * scale_w, val_at(r,1) * scale_h });
    armor_key_points.push_back({ val_at(r,6) * scale_w, val_at(r,7) * scale_h });
    armor_key_points.push_back({ val_at(r,4) * scale_w, val_at(r,5) * scale_h });
    armor_key_points.push_back({ val_at(r,2) * scale_w, val_at(r,3) * scale_h });

    float min_x = armor_key_points[0].x, max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y, max_y = armor_key_points[0].y;
    for (size_t i = 1; i < armor_key_points.size(); ++i) {
      min_x = std::min(min_x, armor_key_points[i].x);
      max_x = std::max(max_x, armor_key_points[i].x);
      min_y = std::min(min_y, armor_key_points[i].y);
      max_y = std::max(max_y, armor_key_points[i].y);
    }

    boxes.emplace_back(static_cast<int>(min_x), static_cast<int>(min_y),
                       static_cast<int>(std::max(0.f, max_x - min_x)),
                       static_cast<int>(std::max(0.f, max_y - min_y)));
    color_ids.emplace_back(color_id.x);
    num_ids.emplace_back(class_id.x);
    confidences.emplace_back(static_cast<float>(score));
    armors_key_points.emplace_back(armor_key_points);
  }

  std::vector<int> indices;
  if (!boxes.empty()) cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (int idx : indices) {
    if (use_roi_) {
      armors.emplace_back(color_ids[idx], num_ids[idx], confidences[idx], boxes[idx], armors_key_points[idx], offset_);
    } else {
      armors.emplace_back(color_ids[idx], num_ids[idx], confidences[idx], boxes[idx], armors_key_points[idx]);
    }
  }

  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it) || !check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    if (use_traditional_) detector_.detect(*it, bgr_img);
    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);
  return armors;
}

// (postprocess, check_name, check_type, get_center_norm, draw_detections, save, sigmoid)
// (保留你所有这些函数的原始代码)
std::list<Armor> YOLOV5::postprocess(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}
bool YOLOV5::check_name(const Armor & armor) const
{
  return armor.name != ArmorName::not_armor && armor.confidence > min_confidence_;
}
bool YOLOV5::check_type(const Armor & armor) const
{
  if (armor.type == ArmorType::small)
    return armor.name != ArmorName::one && armor.name != ArmorName::base;
  else
    return armor.name != ArmorName::two && armor.name != ArmorName::sentry && armor.name != ArmorName::outpost;
}
cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  return { center.x / static_cast<float>(bgr_img.cols), center.y / static_cast<float>(bgr_img.rows) };
}
void YOLOV5::draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format("{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }
  if (use_roi_) {
    cv::rectangle(detection, roi_, cv::Scalar(0,255,0), 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}
void YOLOV5::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}
double YOLOV5::sigmoid(double x)
{
  if (x > 0) return 1.0 / (1.0 + exp(-x));
  return exp(x) / (1.0 + exp(x));
}

} // namespace auto_aim