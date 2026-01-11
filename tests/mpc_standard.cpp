#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp> // 如果 plotter 不依赖于 GUI，可以在不显示时也保持启用
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

// 修改帮助文本，并添加 display 参数
const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/demo.yaml     | 位置参数，yaml配置文件路径 }"
  "{display d      | false                 | 是否开启图形可视化 (true/false) }"; // 新增参数

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter; 

  cv::CommandLineParser cli(argc, argv, keys);
  
  // 获取配置文件路径
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  // 获取 display 参数，默认为 false
  bool enable_display = cli.get<bool>("display");


  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  // 帧率检测相关变量
  int frame_count = 0;
  auto last_fps_time = std::chrono::steady_clock::now();
  double fps = 0.0;

  std::atomic<bool> quit = false;
  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      auto target = target_queue.front();
      auto gs = gimbal.state();
      auto plan = planner.plan(target, 13.4);

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);

      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_pitch_vel"] = gs.pitch_vel;

      data["target_yaw"] = plan.target_yaw;
      data["target_pitch"] = plan.target_pitch;

      data["plan_yaw"] = plan.yaw;
      data["plan_yaw_vel"] = plan.yaw_vel;
      data["plan_yaw_acc"] = plan.yaw_acc;

      data["plan_pitch"] = plan.pitch;
      data["plan_pitch_vel"] = plan.pitch_vel;
      data["plan_pitch_acc"] = plan.pitch_acc;

      data["fire"] = plan.fire ? 1 : 0;
      data["fired"] = fired ? 1 : 0;

      // 添加帧率数据
      data["fps"] = fps;

      if (target.has_value()) {
        data["target_z"] = target->ekf_x()[4];   //z
        data["target_vz"] = target->ekf_x()[5];  //vz
      }

      if (target.has_value()) {
        data["w"] = target->ekf_x()[7];
      } else {
        data["w"] = 0.0;
      }

      plotter.plot(data); // 假设即使不显示图像，数据记录/绘图也可以运行

      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  while (!exiter.exit()) {
    camera.read(img, t);
    auto q = gimbal.q(t);

    solver.set_R_gimbal2world(q);
    
    // 帧率计算 (无论是否显示都计算)
    frame_count++;
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_time).count();
    
    if (elapsed_time >= 1000) { // 每1秒计算一次帧率
      fps = frame_count * 1000.0 / elapsed_time;
      frame_count = 0;
      last_fps_time = current_time;
      
      // 在控制台输出帧率信息 (始终输出)
      fmt::print("FPS: {:.2f}\n", fps);
    }
    
    // 修改detect调用，传入frame_count
    auto armors = yolo.detect(img, frame_count);
    auto targets = tracker.track(armors, t);
    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    // 图像处理和可视化部分 - 仅在启用显示时执行
    if (enable_display) {
        if (!targets.empty()) {
          auto target = targets.front();

          // 当前帧target更新后
          std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
          for (const Eigen::Vector4d & xyza : armor_xyza_list) {
            auto image_points =
              solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
            tools::draw_points(img, image_points, {0, 255, 0});
          }

          Eigen::Vector4d aim_xyza = planner.debug_xyza;
          auto image_points =
            solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 0, 255});
        }

        // 在图像上显示帧率
        std::string fps_text = fmt::format("FPS: {:.2f}", fps);
        cv::putText(img, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::resize(img, img, {}, 0.5, 0.5);
        cv::imshow("reprojection", img);
        auto key = cv::waitKey(1);
        if (key == 'q') break; 
    }

  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}