#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <list>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>

// TensorRT 头文件
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "tasks/auto_aim/yolos/yolov5.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/logger.hpp"
#include "tasks/auto_aim/armor.hpp"

namespace auto_aim
{
namespace multithread
{

class MultiThreadDetector
{
public:
  MultiThreadDetector(const std::string & config_path, bool debug = false);
  ~MultiThreadDetector();

  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();

  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

  bool empty() const { return queue_.empty(); }
  size_t size() const { return queue_.size(); }

private:
  // TensorRT日志记录器
  class TrtLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override;
  };
  
  // TensorRT引擎管理 - 使用智能指针
  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  
  // CUDA资源
  cudaStream_t stream_;
  void* device_buffers_[2]; // 输入和输出缓冲区
  
  // 模型配置
  std::string model_path_;
  int input_width_;
  int input_height_;
  int input_channels_;
  size_t input_size_;
  size_t output_size_;
  
  YOLO yolo_;
  bool debug_;

  // 线程安全队列实现
  template<typename T>
  class ThreadSafeQueue {
  public:
    ThreadSafeQueue(size_t max_size = 16, std::function<void()> full_callback = nullptr)
      : max_size_(max_size), full_callback_(full_callback) {}

    void push(T value) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (queue_.size() >= max_size_) {
        if (full_callback_) full_callback_();
        queue_.pop();
      }
      queue_.push(std::move(value));
      condition_.notify_one();
    }

    T pop() {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this] { return !queue_.empty(); });
      T value = std::move(queue_.front());
      queue_.pop();
      return value;
    }

    bool empty() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return queue_.empty();
    }

    size_t size() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return queue_.size();
    }

  private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<T> queue_;
    size_t max_size_;
    std::function<void()> full_callback_;
  };

  // 推理任务结构
  struct InferenceTask {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<float> output;
    
    InferenceTask(cv::Mat img_, std::chrono::steady_clock::time_point t_, std::vector<float> output_)
      : img(std::move(img_)), timestamp(t_), output(std::move(output_)) {}
  };

  ThreadSafeQueue<InferenceTask> queue_{16, [] { 
    tools::logger()->debug("[MultiThreadDetector] queue is full!"); 
  }};

  // 私有方法
  bool initTensorRT();
  void cleanupTensorRT();
  cv::Mat preprocessImage(const cv::Mat& img);
  std::vector<float> executeInference(const cv::Mat& preprocessed_img);
  
  // 后处理方法
  std::list<Armor> postprocess(double scale, cv::Mat& output, const cv::Mat& bgr_img, int frame_count);
};

}  // namespace multithread
}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP