#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>
#include <fstream>
#include <stdexcept>

namespace auto_aim
{
namespace multithread
{

// TrtLogger 实现
void MultiThreadDetector::TrtLogger::log(Severity severity, const char* msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    tools::logger()->info("[TENSORRT] {}", msg);
  }
}

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: runtime_(nullptr), engine_(nullptr), context_(nullptr), 
  stream_(nullptr), yolo_(config_path), debug_(debug)
{
  // 初始化设备缓冲区
  device_buffers_[0] = nullptr;
  device_buffers_[1] = nullptr;
  
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  model_path_ = yaml[yolo_name + "_model_path"].as<std::string>();
  
  // 设置输入尺寸
  input_width_ = 640;
  input_height_ = 640;
  input_channels_ = 3;
  input_size_ = input_width_ * input_height_ * input_channels_ * sizeof(float);

  // 初始化TensorRT
  if (!initTensorRT()) {
    tools::logger()->error("Failed to initialize TensorRT detector");
    throw std::runtime_error("TensorRT initialization failed");
  }

  tools::logger()->info("[MultiThreadDetector] TensorRT initialized successfully!");
}

MultiThreadDetector::~MultiThreadDetector()
{
  cleanupTensorRT();
}

bool MultiThreadDetector::initTensorRT()
{
  try {
    // 读取引擎文件
    std::ifstream engine_file(model_path_, std::ios::binary);
    if (!engine_file) {
      tools::logger()->error("Cannot open engine file: {}", model_path_);
      return false;
    }

    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    engine_file.close();

    // 创建runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      tools::logger()->error("Failed to create TensorRT runtime");
      return false;
    }

    // 反序列化引擎
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_size));
    if (!engine_) {
      tools::logger()->error("Failed to deserialize engine");
      return false;
    }

    // 创建执行上下文
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      tools::logger()->error("Failed to create execution context");
      return false;
    }

    // 创建CUDA流
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      tools::logger()->error("Failed to create CUDA stream");
      return false;
    }

    // 使用TensorRT 10.x的新API
    // 获取IO tensor数量
    int num_io_tensors = engine_->getNbIOTensors();
    tools::logger()->info("Number of IO tensors: {}", num_io_tensors);
    
    if (num_io_tensors != 2) {
      tools::logger()->error("Expected 2 IO tensors (input and output), got {}", num_io_tensors);
      return false;
    }

    // 获取输入输出tensor名称
    std::vector<const char*> tensor_names;
    for (int i = 0; i < num_io_tensors; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      tensor_names.push_back(tensor_name);
      tools::logger()->info("Tensor {}: {}", i, tensor_name);
    }

    // 获取tensor形状并计算大小
    nvinfer1::Dims input_dims = context_->getTensorShape(tensor_names[0]);
    nvinfer1::Dims output_dims = context_->getTensorShape(tensor_names[1]);

    // 计算输入大小
    input_size_ = sizeof(float);
    for (int i = 0; i < input_dims.nbDims; ++i) {
      input_size_ *= input_dims.d[i];
    }

    // 计算输出大小
    output_size_ = sizeof(float);
    for (int i = 0; i < output_dims.nbDims; ++i) {
      output_size_ *= output_dims.d[i];
    }

    // 分配GPU内存
    if (cudaMalloc(&device_buffers_[0], input_size_) != cudaSuccess) {
      tools::logger()->error("Failed to allocate GPU memory for input");
      return false;
    }

    if (cudaMalloc(&device_buffers_[1], output_size_) != cudaSuccess) {
      tools::logger()->error("Failed to allocate GPU memory for output");
      return false;
    }

    // 设置tensor地址
    context_->setTensorAddress(tensor_names[0], device_buffers_[0]);
    context_->setTensorAddress(tensor_names[1], device_buffers_[1]);

    tools::logger()->info("TensorRT engine loaded: input_size={}, output_size={}", 
                         input_size_, output_size_);
    return true;

  } catch (const std::exception& e) {
    tools::logger()->error("TensorRT initialization exception: {}", e.what());
    cleanupTensorRT();
    return false;
  }
}

void MultiThreadDetector::cleanupTensorRT()
{
  // 释放GPU内存
  for (int i = 0; i < 2; ++i) {
    if (device_buffers_[i]) {
      cudaFree(device_buffers_[i]);
      device_buffers_[i] = nullptr;
    }
  }

  // 释放CUDA流
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }

  // TensorRT 10.x 使用智能指针自动管理资源
  context_.reset();
  engine_.reset();
  runtime_.reset();
}

cv::Mat MultiThreadDetector::preprocessImage(const cv::Mat& img)
{
  // 计算缩放比例
  auto x_scale = static_cast<double>(input_width_) / img.cols;
  auto y_scale = static_cast<double>(input_height_) / img.rows;
  auto scale = std::min(x_scale, y_scale);
  auto w = static_cast<int>(img.cols * scale);
  auto h = static_cast<int>(img.rows * scale);

  // 调整尺寸并填充
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(w, h));
  
  cv::Mat padded(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(0, 0, w, h)));
  
  // 转换为float并归一化
  cv::Mat float_img;
  padded.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
  
  return float_img;
}

std::vector<float> MultiThreadDetector::executeInference(const cv::Mat& preprocessed_img)
{
  if (!context_ || !device_buffers_[0] || !device_buffers_[1]) {
    throw std::runtime_error("TensorRT not properly initialized");
  }

  // 将输入数据拷贝到GPU
  if (cudaMemcpyAsync(device_buffers_[0], preprocessed_img.data, input_size_, 
                     cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
    throw std::runtime_error("Failed to copy input to GPU");
  }

  // 执行推理 - 使用新的enqueueV3 API
  if (!context_->enqueueV3(stream_)) {
    throw std::runtime_error("Failed to execute inference");
  }

  // 分配输出缓冲区
  std::vector<float> output(output_size_ / sizeof(float));
  
  // 将输出从GPU拷贝回CPU
  if (cudaMemcpyAsync(output.data(), device_buffers_[1], output_size_, 
                     cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
    throw std::runtime_error("Failed to copy output from GPU");
  }

  // 同步流
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    throw std::runtime_error("Failed to synchronize stream");
  }

  return output;
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  try {
    // 预处理图像
    cv::Mat preprocessed_img = preprocessImage(img);
    
    // 执行推理
    auto output = executeInference(preprocessed_img);
    
    // 将结果放入队列
    queue_.push(InferenceTask{img.clone(), t, std::move(output)});
    
  } catch (const std::exception& e) {
    tools::logger()->error("Push failed: {}", e.what());
  }
}

std::list<Armor> MultiThreadDetector::postprocess(double scale, cv::Mat& output, const cv::Mat& bgr_img, int frame_count)
{
  // 简化的后处理实现 - 您需要根据您的YOLO模型输出格式进行调整
  std::list<Armor> armors;
  
  // 这里实现您的后处理逻辑
  // 假设output是YOLO格式的检测结果
  
  return armors;
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  try {
    auto task = queue_.pop();
    
    // 将输出向量转换为OpenCV Mat格式
    int num_detections = static_cast<int>(task.output.size()) / 85;  // YOLOv5格式: 85个值每个检测
    if (num_detections > 0) {
      cv::Mat output_mat(num_detections, 85, CV_32F, task.output.data());
      
      // 计算缩放比例用于后处理
      auto x_scale = static_cast<double>(input_width_) / task.img.rows;
      auto y_scale = static_cast<double>(input_height_) / task.img.cols;
      auto scale = std::min(x_scale, y_scale);
      
      auto armors = postprocess(scale, output_mat, task.img, 0);
      return std::make_tuple(std::move(armors), task.timestamp);
    }
    
    return std::make_tuple(std::list<Armor>{}, task.timestamp);
    
  } catch (const std::exception& e) {
    tools::logger()->error("Pop failed: {}", e.what());
    return std::make_tuple(std::list<Armor>{}, std::chrono::steady_clock::now());
  }
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  try {
    auto task = queue_.pop();
    
    int num_detections = static_cast<int>(task.output.size()) / 85;
    if (num_detections > 0) {
      cv::Mat output_mat(num_detections, 85, CV_32F, task.output.data());
      
      auto x_scale = static_cast<double>(input_width_) / task.img.rows;
      auto y_scale = static_cast<double>(input_height_) / task.img.cols;
      auto scale = std::min(x_scale, y_scale);
      
      auto armors = postprocess(scale, output_mat, task.img, 0);
      return std::make_tuple(task.img, std::move(armors), task.timestamp);
    }
    
    return std::make_tuple(task.img, std::list<Armor>{}, task.timestamp);
    
  } catch (const std::exception& e) {
    tools::logger()->error("Debug pop failed: {}", e.what());
    return std::make_tuple(cv::Mat{}, std::list<Armor>{}, std::chrono::steady_clock::now());
  }
}

}  // namespace multithread
}  // namespace auto_aim