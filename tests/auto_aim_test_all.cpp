#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{@config-path   | configs/demo.yaml | yaml配置文件路径}";

int main(int argc, char * argv[])
{
  // -------------------- 参数读取 --------------------
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  tools::Plotter plotter;
  tools::Exiter exiter;

  // -------------------- 模块初始化 --------------------
  io::Camera camera(config_path);
  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  io::Gimbal gimbal(config_path);

  cv::Mat img;

  // -------------------- FPS 统计 --------------------
  double fps = 0.0;
  auto last_fps_time = std::chrono::steady_clock::now();
  int frame_count = 0;
  constexpr double fps_update_interval = 1.0;

  // -------------------- 主循环 --------------------
  while (!exiter.exit()) {
    auto frame_start = std::chrono::steady_clock::now();

    camera.read(img, frame_start);
    if (img.empty()) break;

    // -------------------- 云台姿态 --------------------
    Eigen::Quaterniond gimbal_q = gimbal.q(frame_start);
    solver.set_R_gimbal2world(gimbal_q);
    Eigen::Vector3d ypr = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0);
    Eigen::Vector2d current_ypr = ypr.head<2>();
    
    // -------------------- 自瞄主流程 --------------------
    auto armors = yolo.detect(img, 0);
    auto targets = tracker.track(armors, frame_start);
    auto command = aimer.aim(targets, frame_start, 13, false);

    // -------------------- 串口发送 --------------------
    gimbal.send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);

    // -------------------- FPS 计算 --------------------
    frame_count++;
    auto now = std::chrono::steady_clock::now();
    double elapsed = tools::delta_time(now, last_fps_time);

    if (elapsed >= fps_update_interval) {
      fps = frame_count / elapsed;
      frame_count = 0;
      last_fps_time = now;

      tools::logger()->info("FPS: {:.1f}, Tracker State: {}", fps, tracker.state());
    }

    // -------------------- 基础文字信息 --------------------
    Eigen::Vector3d euler = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3;

    std::string tracker_state = tracker.state();
    cv::Scalar state_color;
    if (tracker_state == "tracking")
      state_color = {0, 255, 0};
    else if (tracker_state == "temp_lost")
      state_color = {0, 255, 255};
    else if (tracker_state == "detecting")
      state_color = {255, 255, 0};
    else
      state_color = {0, 0, 255};

    tools::draw_text(img, fmt::format("State: {}", tracker_state), {10, 30}, state_color);
    tools::draw_text(
      img, fmt::format("Gimbal yaw:{:.2f}, pitch:{:.2f}", euler[0], euler[1]), {10, 60},
      {255, 255, 255});
    tools::draw_text(
      img, fmt::format("Cmd yaw:{:.2f}, pitch:{:.2f}", command.yaw * 57.3, command.pitch * 57.3),
      {10, 90}, {154, 50, 205});

    if (command.shoot) {
      tools::draw_text(img, "FIRE", {10, 120}, cv::Scalar(0, 0, 255), 1.2);
    } else {
      tools::draw_text(img, "HOLD", {10, 120}, cv::Scalar(200, 200, 200));
    }

    // ==================== 【关键修改：可视化】====================

    // 1. 绘制所有装甲板位置（绿色）
    for (const auto & target : targets) {
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points = solver.reproject_armor(xyza.head<3>(), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }
    }

    // 2. 绘制瞄准点（单个红点）
    if (!targets.empty() && aimer.debug_aim_point.valid) {
      Eigen::Vector3d aim_xyz(
        aimer.debug_aim_point.xyza[0],
        aimer.debug_aim_point.xyza[1],
        aimer.debug_aim_point.xyza[2]
      );

      // 转为 OpenCV 点格式
      std::vector<cv::Point3f> world_pts = {
        cv::Point3f(static_cast<float>(aim_xyz.x()),
                    static_cast<float>(aim_xyz.y()),
                    static_cast<float>(aim_xyz.z()))
      };

      // 投影到图像平面
      auto pixel_pts = solver.world2pixel(world_pts);

      if (!pixel_pts.empty()) {
        cv::Point2f pt = pixel_pts[0];
        // 实心红圆 + 白边（更醒目）
        cv::circle(img, pt, 5, cv::Scalar(0, 0, 255), -1);      // 红色填充
        cv::circle(img, pt, 6, cv::Scalar(255, 255, 255), 1);   // 白色轮廓
      }
    }

    // ==================== 【End 可视化】====================

    // -------------------- Plotter --------------------
    nlohmann::json data;
    data["fps"] = fps;
    data["gimbal_yaw"] = euler[0];
    data["gimbal_pitch"] = euler[2];
    data["cmd_yaw"] = command.yaw * 57.3;
    data["cmd_pitch"] = command.pitch * 57.3;
    data["track_state"] = tracker_state;
    data["control"] = command.control;

    plotter.plot(data);

    // -------------------- 显示 --------------------
    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("auto_aim", img);
    if (cv::waitKey(1) == 'q') break;
  }

  return 0;
}