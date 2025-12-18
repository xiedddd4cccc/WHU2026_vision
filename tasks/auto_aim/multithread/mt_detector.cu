#include "mt_detector.hpp" // [修改] 包含新头文件

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <cstring>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include <opencv2/dnn.hpp>
#include <cuda_fp16.h> 

#include <opencv2/cudawarping.hpp> 
#include <opencv2/core/cuda_stream_accessor.hpp> 


// [不变] CUDA Kernel
static __global__
void bgr_to_rgb_nchw_kernel_float(const float* __restrict__ src, size_t src_step_bytes, void* dst_void,
                                  int H, int W, bool dst_fp16)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;

  const char* row_ptr_char = reinterpret_cast<const char*>(src) + y * src_step_bytes;
  const float* row_ptr = reinterpret_cast<const float*>(row_ptr_char);
  int idx = x * 3;
  float b = row_ptr[idx + 0];
  float g = row_ptr[idx + 1];
  float r = row_ptr[idx + 2];

  int hw = H * W;
  int pix = y * W + x;

  if (!dst_fp16) {
    float* dst = reinterpret_cast<float*>(dst_void);
    dst[0 * hw + pix] = r; 
    dst[1 * hw + pix] = g;
    dst[2 * hw + pix] = b;
  } else {
    __half* dst = reinterpret_cast<__half*>(dst_void);
    dst[0 * hw + pix] = __float2half_rn(r);
    dst[1 * hw + pix] = __float2half_rn(g);
    dst[2 * hw + pix] = __float2half_rn(b);
  }
}

namespace auto_aim
{

// ---------------- Constructor / Destructor ----------------
// [修改] 重命名为 MTDetector
MTDetector::MTDetector(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  try {
    auto yaml = YAML::LoadFile(config_path);
    
    // [关键] 确保 YAML 键名与你的 mt_detector 匹配
    auto yolo_name = yaml["yolo_name"].as<std::string>();
    model_path_ = yaml[yolo_name + "_model_path"].as<std::string>();
    
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
    if (yaml["use_gpu_preproc"]) use_gpu_preproc_ = yaml["use_gpu_preproc"].as<bool>();
    save_path_ = "imgs";
    std::filesystem::create_directory(save_path_);
    std::ifstream engineFile(model_path_, std::ios::binary);
    if (!engineFile.is_open()) { /*...*/ return; }
    engineFile.seekg(0, std::ios::end);
    std::streamsize fsize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(static_cast<size_t>(fsize));
    if (!engineFile.read(engineData.data(), fsize)) { /*...*/ return; }
    engineFile.close();
    nvinfer1::IRuntime* runtime_ptr = nvinfer1::createInferRuntime(mLogger_);
    if (!runtime_ptr) { /*...*/ return; }
    mRuntime.reset(runtime_ptr);
    nvinfer1::ICudaEngine* engine_ptr = mRuntime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine_ptr) { /*...*/ return; }
    mEngine.reset(engine_ptr);
    
    // [恢复] 只创建 1 个 Context
    mContext.reset(mEngine->createExecutionContext());
    if (!mContext) {
      tools::logger()->error("MTDetector: createExecutionContext failed");
      return;
    }

    // [恢复] 只创建 1 个 Stream
    CHECK_CUDA(cudaStreamCreate(&mStream));

    allocateBindings(); // (已重写)
    tools::logger()->info("MTDetector: TensorRT engine loaded. Input: {} Output: {}",
                          mInputTensorName_, mOutputTensorName_);
  }
  catch (const std::exception & e) {
    tools::logger()->error("MTDetector: Exception in ctor: {}", e.what());
    throw std::runtime_error("Failed during MTDetector ctor: " + std::string(e.what()));
  }
}

// [修改] 重命名为 MTDetector
MTDetector::~MTDetector()
{
  try {
    tools::logger()->info("MTDetector: Releasing resources...");
    freeBindings(); 
    freePinnedBuffers(); 

    if (mStream) CHECK_CUDA(cudaStreamDestroy(mStream));

    tools::logger()->info("MTDetector: All resources released.");
  } catch (...) {}
}

// --------------- allocate / free bindings ---------------
void MTDetector::allocateBindings()
{
  if (!mEngine) return;
  int nbIOTensors = mEngine->getNbIOTensors();
  if (nbIOTensors <= 0) { /*...*/ return; }
  
  size_t elems_in = 0;
  size_t elems_out = 0;

  for (int i = 0; i < nbIOTensors; ++i) {
    const char* tname = mEngine->getIOTensorName(i);
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
      elems_in = elems;

      // [恢复] 只分配 1 套 TRT 输入缓冲区
      CHECK_CUDA(cudaMalloc(&mGpuBuffers_[0], bytes));
      tools::logger()->info("MTDetector: Alloc TRT input buffer ({} bytes)", bytes);
    
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      mOutputTensorName_ = name;
      mOutputDims_ = dims;
      mOutputBytes_ = bytes; 
      elems_out = elems;

      // [恢复] 只分配 1 套 TRT 输出缓冲区
      CHECK_CUDA(cudaMalloc(&mGpuBuffers_[1], bytes));
      mCpuOutputBuffer_.resize(elems_out);
      tools::logger()->info("MTDetector: Alloc TRT output buffer ({} bytes)", bytes);
    }
  }
  initPinnedBuffers();
}
void MTDetector::freeBindings()
{
  if (mGpuBuffers_[0]) CHECK_CUDA(cudaFree(mGpuBuffers_[0]));
  if (mGpuBuffers_[1]) CHECK_CUDA(cudaFree(mGpuBuffers_[1]));
  mGpuBuffers_[0] = nullptr;
  mGpuBuffers_[1] = nullptr;
  mCpuOutputBuffer_.clear();
}
void MTDetector::initPinnedBuffers()
{
  if (mPinnedOutputHost_) return;
  if (mOutputBytes_ == 0) { /*...*/ return; }
  mPinnedOutputBytes_ = mOutputBytes_;
  
  // [恢复] 只分配 1 套 D2H 固定内存
  CHECK_CUDA(cudaHostAlloc(&mPinnedOutputHost_, mPinnedOutputBytes_, cudaHostAllocDefault));
  
  if (debug_) tools::logger()->info("MTDetector: 1 pinned output buffer allocated {} bytes", mPinnedOutputBytes_);
  
  size_t elems = mPinnedOutputBytes_ / sizeof(float);
  mCpuOutputBuffer_.resize(elems);
}
void MTDetector::freePinnedBuffers()
{
  if (mPinnedOutputHost_) { cudaFreeHost(mPinnedOutputHost_); mPinnedOutputHost_ = nullptr; }
}
void MTDetector::gpuPreprocessToDevice(const cv::Mat &src_bgr, void* device_input_ptr, int inW, int inH, nvinfer1::DataType dtype, cudaStream_t stream)
{
  auto cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
  cv::cuda::GpuMat d_src;
  d_src.upload(src_bgr, cv_stream);
  cv::cuda::GpuMat d_resized;
  cv::cuda::resize(d_src, d_resized, cv::Size(inW, inH), 0, 0, cv::INTER_LINEAR, cv_stream);
  cv::cuda::GpuMat d_float;
  d_resized.convertTo(d_float, CV_32F, 1.0 / 255.0, 0.0, cv_stream);
  dim3 block(16, 16);
  dim3 grid((inW + block.x - 1) / block.x, (inH + block.y - 1) / block.y);
  bool dst_fp16 = (dtype == nvinfer1::DataType::kHALF);
  float* d_float_ptr = reinterpret_cast<float*>(d_float.data);
  size_t src_step_bytes = d_float.step;
  bgr_to_rgb_nchw_kernel_float<<<grid, block, 0, stream>>>(d_float_ptr, src_step_bytes, device_input_ptr, inH, inW, dst_fp16);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    tools::logger()->error("MTDetector: gpu preprocess kernel error: {}", cudaGetErrorString(err)); // [修改] 日志
  }
}
// ...


// ----------------- detect -----------------
// -----------------------------------------------------------------
// ⬇️ ⬇️ [关键修改] 恢复为“串行同步”的 detect ⬇️ ⬇️
// -----------------------------------------------------------------
std::list<Armor> MTDetector::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    if (debug_) tools::logger()->warn("MTDetector: detect empty image");
    return {};
  }
  cv::Mat bgr_img = use_roi_ ? raw_img(roi_) : raw_img;
  if (mInputDims_.nbDims < 4) {
    tools::logger()->error("MTDetector: invalid input dims");
    return {};
  }
  
  int inN = mInputDims_.d[0] > 0 ? mInputDims_.d[0] : 1;
  int inC = mInputDims_.d[1];
  int inH = mInputDims_.d[2];
  int inW = mInputDims_.d[3];

  // --- 1. GPU 预处理 (异步) ---
  if (use_gpu_preproc_ && mGpuBuffers_[0]) {
    gpuPreprocessToDevice(bgr_img, mGpuBuffers_[0], inW, inH, mInputDtype_, mStream);
  } else {
    // (fallback CPU 预处理)
    cv::Mat resized;
    cv::resize(bgr_img, resized, cv::Size(inW, inH));
    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, cv::Size(inW, inH), cv::Scalar(), true, false, CV_32F);
    cv::Mat input_contig = blob.isContinuous() ? blob : blob.clone();
    CHECK_CUDA(cudaMemcpyAsync(mGpuBuffers_[0], input_contig.data, std::min(input_contig.total() * input_contig.elemSize(), mInputBytes_), cudaMemcpyHostToDevice, mStream));
  }

  // --- 2. GPU 推理 (异步) ---
  mContext->setTensorAddress(mInputTensorName_.c_str(), mGpuBuffers_[0]);
  mContext->setTensorAddress(mOutputTensorName_.c_str(), mGpuBuffers_[1]);
  if (!mContext->enqueueV3(mStream)) {
    tools::logger()->error("MTDetector: enqueueV3 failed");
    return {};
  }

  // --- 3. D2H 拷贝 (异步) ---
  size_t to_copy_back = mPinnedOutputBytes_ ? std::min(mPinnedOutputBytes_, mOutputBytes_) : mOutputBytes_;
  CHECK_CUDA(cudaMemcpyAsync(mPinnedOutputHost_, mGpuBuffers_[1], to_copy_back, cudaMemcpyDeviceToHost, mStream));
  
  // --- 4. 同步 ---
  // [关键] CPU 在此等待 (12ms)，直到 1, 2, 3 全部完成
  CHECK_CUDA(cudaStreamSynchronize(mStream));

  // --- 5. CPU 拷贝 (Pinned -> Pageable) ---
  size_t floats = to_copy_back / sizeof(float);
  if (mCpuOutputBuffer_.size() < floats) mCpuOutputBuffer_.resize(floats);
  std::memcpy(mCpuOutputBuffer_.data(), mPinnedOutputHost_, floats * sizeof(float));

  // --- 6. CPU 后处理 ---
  int out_rows = 1;
  int out_cols = static_cast<int>(mCpuOutputBuffer_.size());
  // ... (reshape 逻辑) ...
  if (mOutputDims_.nbDims >= 3) {
      out_rows = mOutputDims_.d[mOutputDims_.nbDims - 2];
      out_cols = mOutputDims_.d[mOutputDims_.nbDims - 1];
  } else if (mOutputDims_.nbDims == 2) {
      out_rows = mOutputDims_.d[0];
      out_cols = mOutputDims_.d[1];
  }
  if (static_cast<size_t>(out_rows * out_cols) != mCpuOutputBuffer_.size() && mCpuOutputBuffer_.size() > 0) {
    if (debug_) tools::logger()->warn("MTDetector: Output dims [{}, {}] do not match total elements {}.",
                                  out_rows, out_cols, mCpuOutputBuffer_.size());
    if (mOutputDims_.nbDims >= 2 && mOutputDims_.d[mOutputDims_.nbDims - 1] > 0) {
        out_cols = mOutputDims_.d[mOutputDims_.nbDims - 1];
        out_rows = mCpuOutputBuffer_.size() / out_cols;
    }
    if (static_cast<size_t>(out_rows * out_cols) != mCpuOutputBuffer_.size()) {
       tools::logger()->error("MTDetector: Cannot reshape output buffer. Using [1, total_elements].");
       out_rows = 1;
       out_cols = static_cast<int>(mCpuOutputBuffer_.size());
    }
  }
  cv::Mat output(out_rows, out_cols, CV_32F, mCpuOutputBuffer_.data());
  
  std::list<Armor> armors = parse(1.0, output, bgr_img, frame_count);

  if (debug_) {
    draw_detections(bgr_img, armors, frame_count);
  }
  
  return armors;
}

// ----------------- parse / postprocess / helpers -----------------
// -----------------------------------------------------------------
// ⬇️ ⬇️ [关键] 你 *必须* 为你的 MTDetector 模型重写这个函数 ⬇️ ⬇️
// (下面的代码是 YOLOV5 的占位符，它很可能会出错)
// -----------------------------------------------------------------
std::list<Armor> MTDetector::parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  tools::logger()->warn("MTDetector::parse() is using the YOLOV5 placeholder. Please update me!");

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
      tools::logger()->error("MTDetector::parse: Invalid output mat.");
      return {};
  }
  
  // (YOLOV5 的解析逻辑，用于占位)
  for (int r = 0; r < output.rows; ++r) {
    if (output.cols < 22) {
        if(r == 0 && debug_) tools::logger()->warn("MTDetector::parse: Output cols ({}) < 22. May truncate.", output.cols);
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
    
    armor_key_points.push_back({ val_at(r,0) * (float)scale_w, val_at(r,1) * (float)scale_h });
    armor_key_points.push_back({ val_at(r,6) * (float)scale_w, val_at(r,7) * (float)scale_h });
    armor_key_points.push_back({ val_at(r,4) * (float)scale_w, val_at(r,5) * (float)scale_h });
    armor_key_points.push_back({ val_at(r,2) * (float)scale_w, val_at(r,3) * (float)scale_h });

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
  
  return armors;
}

// (其他辅助函数，保持不变)
std::list<Armor> MTDetector::postprocess(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}
bool MTDetector::check_name(const Armor & armor) const
{
  return armor.name != ArmorName::not_armor && armor.confidence > min_confidence_;
}
bool MTDetector::check_type(const Armor & armor) const
{
  if (armor.type == ArmorType::small)
    return armor.name != ArmorName::one && armor.name != ArmorName::base;
  else
    return armor.name != ArmorName::two && armor.name != ArmorName::sentry && armor.name != ArmorName::outpost;
}
cv::Point2f MTDetector::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  return { center.x / static_cast<float>(bgr_img.cols), center.y / static_cast<float>(bgr_img.rows) };
}
void MTDetector::draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  if (debug_) {
    tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  }
  for (const auto & armor : armors) {
    auto info = fmt::format("{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    if (debug_) tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }
  if (use_roi_) {
    cv::rectangle(detection, roi_, cv::Scalar(0,255,0), 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}
void MTDetector::save(const Armor & armor) const
{
  if (tmp_img_.empty()) return;
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}
double MTDetector::sigmoid(double x)
{
  if (x > 0) return 1.0 / (1.0 + exp(-x));
  return exp(x) / (1.0 + exp(x));
}

} // namespace auto_aim