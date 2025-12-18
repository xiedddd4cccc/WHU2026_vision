#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明}"
  "{@config-path   | configs/demo.yaml | yaml配置文件路径}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>("@config-path");

  tools::Exiter exiter;
  tools::Plotter plotter;
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  auto_aim::multithread::MultiThreadDetector detector(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  // 性能计时变量
  auto last_frame_time = std::chrono::steady_clock::now();
  int frame_count = 0;
  double fps = 0.0;
  constexpr double fps_update_interval = 1.0;

  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
      auto read_start = std::chrono::steady_clock::now();
      camera.read(img, t);
      auto read_end = std::chrono::steady_clock::now();
      
      double read_time = tools::delta_time(read_end, read_start) * 1000;
      tools::logger()->debug("Camera read time: {:.2f}ms", read_time);
      
      detector.push(img, t);
    }
  });

  auto last_t = std::chrono::steady_clock::now();
  nlohmann::json data;

  while (!exiter.exit()) {
    auto frame_start = std::chrono::steady_clock::now();
    
    // 从检测器获取数据
    auto pop_start = std::chrono::steady_clock::now();
    auto [img, armors, t] = detector.debug_pop();
    auto pop_end = std::chrono::steady_clock::now();
    double pop_time = tools::delta_time(pop_end, pop_start) * 1000;

    auto process_start = std::chrono::steady_clock::now();

    // 使用Gimbal获取姿态数据
    auto gimbal_start = std::chrono::steady_clock::now();
    Eigen::Quaterniond q = gimbal.q(t);
    solver.set_R_gimbal2world(q);
    auto gimbal_end = std::chrono::steady_clock::now();
    double gimbal_time = tools::delta_time(gimbal_end, gimbal_start) * 1000;

    Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    // 跟踪
    auto track_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, t);
    auto track_end = std::chrono::steady_clock::now();
    double track_time = tools::delta_time(track_end, track_start) * 1000;

    // 瞄准
    auto aim_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, t, 22);
    auto aim_end = std::chrono::steady_clock::now();
    double aim_time = tools::delta_time(aim_end, aim_start) * 1000;

    // 发送控制指令
    auto send_start = std::chrono::steady_clock::now();
    gimbal.send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);
    auto send_end = std::chrono::steady_clock::now();
    double send_time = tools::delta_time(send_end, send_start) * 1000;

    auto process_end = std::chrono::steady_clock::now();
    double process_time = tools::delta_time(process_end, process_start) * 1000;

    auto dt = tools::delta_time(t, last_t);
    last_t = t;

    // 帧率计算
    frame_count++;
    double elapsed_time = tools::delta_time(std::chrono::steady_clock::now(), last_frame_time);
    if (elapsed_time >= fps_update_interval) {
      fps = frame_count / elapsed_time;
      frame_count = 0;
      last_frame_time = std::chrono::steady_clock::now();
    }

    // 总帧时间
    auto frame_end = std::chrono::steady_clock::now();
    double total_frame_time = tools::delta_time(frame_end, frame_start) * 1000;

    // 输出详细的性能日志
    tools::logger()->info(
      "Frame {:4d} | Total: {:5.1f}ms | FPS: {:5.1f} | "
      "Pop: {:4.1f}ms | Gimbal: {:4.1f}ms | Track: {:4.1f}ms | "
      "Aim: {:4.1f}ms | Send: {:4.1f}ms | Process: {:4.1f}ms | "
      "Armors: {:2d} | State: {} | Shoot: {}",
      frame_count, total_frame_time, fps,
      pop_time, gimbal_time, track_time,
      aim_time, send_time, process_time,
      armors.size(), tracker.state(), command.shoot);

    // 每10帧输出一次详细的时间分布（避免日志过多）
    static int detail_log_counter = 0;
    if (detail_log_counter++ % 10 == 0) {
      tools::logger()->debug(
        "Time distribution - Pop: {:.1f}% Gimbal: {:.1f}% Track: {:.1f}% Aim: {:.1f}% Send: {:.1f}% Other: {:.1f}%",
        (pop_time / total_frame_time) * 100,
        (gimbal_time / total_frame_time) * 100,
        (track_time / total_frame_time) * 100,
        (aim_time / total_frame_time) * 100,
        (send_time / total_frame_time) * 100,
        ((total_frame_time - pop_time - gimbal_time - track_time - aim_time - send_time) / total_frame_time) * 100);
    }

    data["dt"] = dt;
    data["fps"] = 1 / dt;
    data["total_frame_time"] = total_frame_time;
    data["pop_time"] = pop_time;
    data["gimbal_time"] = gimbal_time;
    data["track_time"] = track_time;
    data["aim_time"] = aim_time;
    data["send_time"] = send_time;
    data["process_time"] = process_time;
    
    // 装甲板原始观测数据
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      auto min_x = 1e10;
      auto & armor = armors.front();
      for (auto & a : armors) {
        if (a.center.x < min_x) {
          min_x = a.center.x;
          armor = a;
        }
      }
      solver.solve(armor);
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
    }

    if (!targets.empty()) {
      auto target = targets.front();
      
      // 在图像上显示处理时间信息
      int text_y = 30;
      int line_height = 25;
      
      // 显示帧率和总处理时间
      tools::draw_text(img, fmt::format("FPS: {:.1f} | Total: {:.1f}ms", fps, total_frame_time), 
                       {10, text_y}, {255, 255, 255});
      text_y += line_height;
      
      // 显示跟踪状态
      tools::draw_text(img, fmt::format("State: {}", tracker.state()), 
                       {10, text_y}, {255, 255, 255});
      text_y += line_height;
      
      // 显示各阶段时间
      tools::draw_text(img, fmt::format("Track: {:.1f}ms Aim: {:.1f}ms", track_time, aim_time), 
                       {10, text_y}, {255, 255, 255});
      text_y += line_height;
      
      // 显示装甲板数量和控制状态
      tools::draw_text(img, fmt::format("Armors: {} Shoot: {}", armors.size(), command.shoot), 
                       {10, text_y}, {255, 255, 255});
      text_y += line_height;

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // aimer瞄准位置
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid)
        tools::draw_points(img, image_points, {0, 0, 255});  // red
      else
        tools::draw_points(img, image_points, {255, 0, 0});  // blue

      // 观测器内部数据
      Eigen::VectorXd x = target.ekf_x();
      
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
      data["distance"] = std::sqrt(x[0] * x[0] + x[2] * x[2] + x[4] * x[4]);

      // 卡方检验数据
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
    }
    
    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  detect_thread.join();
  plotter.plot(data);

  return 0;
}