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
#include "tools/recorder.hpp"
using namespace std::chrono_literals;
// ==================== 命令行参数 ====================
const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{@config-path   | configs/demo.yaml | yaml配置文件路径}"
  "{no-vis         | false             | 关闭所有可视化（draw / plot / imshow）}";

int main(int argc, char * argv[])
{
  // -------------------- 参数读取 --------------------
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>(0);
  bool enable_vis = !cli.get<bool>("no-vis");

  tools::Plotter plotter;
  tools::Exiter exiter;

  // -------------------- 模块初始化 --------------------
  io::Camera camera(config_path);
  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  io::Gimbal gimbal(config_path);
  //tools::Recorder recorder(30.0);
  cv::Mat img;

  // -------------------- FPS 统计 --------------------
  double fps = 0.0;
  auto last_fps_time = std::chrono::steady_clock::now();
  int frame_count = 0;
  constexpr double fps_update_interval = 1.0;

  // ==================== 主循环 ====================
  while (!exiter.exit()) {
    auto frame_start = std::chrono::steady_clock::now();

    camera.read(img, frame_start);
    if (img.empty()) break;

    // -------------------- 云台姿态 --------------------
    constexpr auto IMU_DELAY = 2ms;
    Eigen::Quaterniond gimbal_q = gimbal.q(frame_start-IMU_DELAY);
    auto gs=gimbal.state();
    //recorder.record(img, gimbal_q, frame_start);
    solver.set_R_gimbal2world(gimbal_q);
    Eigen::Vector3d ypr =
      tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0);

    // -------------------- 自瞄主流程 --------------------
    auto armors = yolo.detect(img, 0);
    auto targets = tracker.track(armors, frame_start);
    auto command = aimer.aim(targets, frame_start, gs.bullet_speed, true, ypr);

    // -------------------- 串口发送 --------------------
    gimbal.send(
      command.control,
      command.shoot,
      command.yaw,
      0, 0,
      command.pitch,
      0, 0
    );

    // -------------------- FPS 计算 --------------------
    frame_count++;
    auto now = std::chrono::steady_clock::now();
    double elapsed = tools::delta_time(now, last_fps_time);

    if (elapsed >= fps_update_interval) {
      fps = frame_count / elapsed;
      frame_count = 0;
      last_fps_time = now;

      tools::logger()->info(
        "FPS: {:.1f}, Tracker State: {}",
        fps,
        tracker.state()
      );
    }

    // ==================== 可视化（统一开关） ====================
    if (enable_vis) {
      // -------- 基础状态 --------
      Eigen::Vector3d euler =
        tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3;

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

      tools::draw_text(
        img,
        fmt::format("State: {}", tracker_state),
        {10, 30},
        state_color
      );

      tools::draw_text(
        img,
        fmt::format(
          "Gimbal yaw:{:.2f}, pitch:{:.2f}",
          euler[0],
          euler[1]
        ),
        {10, 60},
        {255, 255, 255}
      );

      tools::draw_text(
        img,
        fmt::format(
          "Cmd yaw:{:.2f}, pitch:{:.2f}",
          command.yaw * 57.3,
          command.pitch * 57.3
        ),
        {10, 90},
        {77, 255, 100}
      );

      tools::draw_text(
        img,
        fmt::format("Bullet_speed: {:.3f}", gs.bullet_speed),
        {10, 120},
        {77, 255, 100}
      );

      // -------- Debug：装甲板 & id --------
      for (const auto & target : targets) {
        const auto & armor_xyza_list = target.armor_xyza_list();

        for (size_t i = 0; i < armor_xyza_list.size(); ++i) {
          const Eigen::Vector4d & xyza = armor_xyza_list[i];

          auto image_points = solver.reproject_armor(
            xyza.head<3>(),
            xyza[3],
            target.armor_type,
            target.name
          );
          tools::draw_points(img, image_points, {0, 255, 0});

          auto pixel_pts = solver.world2pixel({
            cv::Point3f(xyza[0], xyza[1], xyza[2])
          });

          if (!pixel_pts.empty()) {
            cv::putText(
              img,
              fmt::format("id={}", i),
              pixel_pts[0] + cv::Point2f(5, -5),
              cv::FONT_HERSHEY_SIMPLEX,
              0.6,
              cv::Scalar(0, 0, 255),
              2
            );
          }
        }
      }

      // -------- Debug：瞄准点 --------
      if (!targets.empty() && aimer.debug_aim_point.valid) {
        auto pixel_pts = solver.world2pixel({
          cv::Point3f(
            aimer.debug_aim_point.xyza[0],
            aimer.debug_aim_point.xyza[1],
            aimer.debug_aim_point.xyza[2]
          )
        });

        if (!pixel_pts.empty()) {
          cv::circle(img, pixel_pts[0], 5, {0, 0, 255}, -1);
          cv::circle(img, pixel_pts[0], 6, {255, 255, 255}, 1);
        }
      }

      // -------- Plotter --------
      nlohmann::json data;
      if(!targets.empty())
      {
        auto target =targets.front();
        Eigen::VectorXd x =target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
      data["a"] = x[6] * 57.3;
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;
      } 
      data["fps"] = fps;
      data["gimbal_yaw"] = euler[0];
      data["gimbal_pitch"] = euler[1];
      data["cmd_yaw"] = command.yaw * 57.3;
      data["cmd_pitch"] = command.pitch * 57.3;
      data["track_state"] = tracker_state;
      data["control"] = command.control;
      data["fire"]=command.shoot;
      data["speed"]=gs.bullet_speed;

      plotter.plot(data);

      // -------- 显示 --------
      cv::resize(img, img, {}, 0.5, 0.5);
      cv::imshow("auto_aim", img);
      if (cv::waitKey(1) == 'q') break;
    }
  }

  return 0;
}
