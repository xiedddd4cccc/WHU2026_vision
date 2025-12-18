#include <fmt/core.h>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"  // 引入Planner
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/shooter.hpp"
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
  auto_aim::Planner planner(config_path);  // 替换Aimer为Planner
  auto_aim::Shooter shooter(config_path);
  io::Gimbal gimbal(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  // 帧率计算变量
  double fps = 0.0;
  auto last_frame_time = std::chrono::steady_clock::now();
  int frame_count = 0;
  constexpr double fps_update_interval = 0.5;
  auto last_fps_update_time = last_frame_time;

  // Planner相关变量
  std::optional<auto_aim::Target> current_target = std::nullopt;
  auto_aim::Plan current_plan;
  bool planner_initialized = false;
  
  // 防丢帧预测变量
  Eigen::Vector2d last_valid_command = Eigen::Vector2d::Zero();  // [yaw, pitch]
  Eigen::Vector2d last_valid_vel = Eigen::Vector2d::Zero();      // [yaw_vel, pitch_vel]
  std::chrono::steady_clock::time_point last_valid_time;
  int consecutive_lost_frames = 0;
  constexpr int MAX_LOST_FRAMES = 5;

  // -------------------- 主循环 --------------------
  while (!exiter.exit()) {
    auto frame_start = std::chrono::steady_clock::now();

    camera.read(img, timestamp);
    if (img.empty()) break;

    Eigen::Quaterniond gimbal_q = gimbal.q(timestamp);
    solver.set_R_gimbal2world(gimbal_q);

    // -------------------- 自瞄核心（使用Planner） --------------------
    auto armors = yolo.detect(img, 0);
    auto targets = tracker.track(armors, timestamp);
    
    // 更新当前目标
    if (!targets.empty()) {
      current_target = targets.front();
      consecutive_lost_frames = 0;
    } else {
      consecutive_lost_frames++;
      if (consecutive_lost_frames > MAX_LOST_FRAMES) {
        current_target = std::nullopt;
      }
    }

    // 使用Planner进行运动规划
    io::Command command{};
    std::string planner_status = "NO_TARGET";
    
    if (current_target.has_value()) {
      try {
        // 使用Planner生成优化轨迹
        current_plan = planner.plan(current_target, 12.0);
        
        if (current_plan.control) {
          command.control = true;
          command.yaw = current_plan.yaw;
          command.pitch = current_plan.pitch;
          
          // 更新有效命令和速度
          last_valid_command << command.yaw, command.pitch;
          last_valid_vel << current_plan.yaw_vel, current_plan.pitch_vel;
          last_valid_time = std::chrono::steady_clock::now();
          planner_initialized = true;
          
          planner_status = "TRACKING";
        }
      } catch (const std::exception& e) {
        tools::logger()->warn("Planner error: {}", e.what());
        planner_status = "PLANNER_ERROR";
      }
    } else if (planner_initialized && consecutive_lost_frames <= MAX_LOST_FRAMES) {
      // 短期预测：使用Planner的运动模型进行预测
      double predict_time = consecutive_lost_frames * 0.013; 
      predict_time = std::min(predict_time, 0.15); // 最大预测150ms
      
      command.control = true;
      command.yaw = last_valid_command[0] + last_valid_vel[0] * predict_time;
      command.pitch = last_valid_command[1] + last_valid_vel[1] * predict_time;
      
      planner_status = fmt::format("PREDICTING({}/{})", consecutive_lost_frames, MAX_LOST_FRAMES);
    } else {
      command.control = false;
      command.yaw = 0;
      command.pitch =0;
      planner_status = "LOST";
    }

    // ==================== 射击判断（基于Planner的优化） ====================
    Eigen::Vector3d gimbal_euler = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0);
    
    bool shooter_ready = false;
    if (current_target.has_value() && current_plan.control) {
      // 使用Planner的射击决策（更智能）
      shooter_ready = current_plan.fire && 
                     tracker.state() == "tracking" && 
                     consecutive_lost_frames == 0;
    }
    
    command.shoot = shooter_ready;

    // -------------------- 串口发送控制命令 --------------------
    if (command.control) {
      tools::logger()->info(
        "Planner[{}] - yaw: {:.2f}°(vel: {:.1f}), pitch: {:.2f}°(vel: {:.1f}), shoot: {}", 
        planner_status, 
        command.yaw * 57.3, current_plan.yaw_vel * 57.3,
        command.pitch * 57.3, current_plan.pitch_vel * 57.3,
        command.shoot);
    }

    gimbal.send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);

    // -------------------- 帧率计算 --------------------
    auto frame_end = std::chrono::steady_clock::now();
    frame_count++;
    
    double elapsed_time = tools::delta_time(frame_end, last_fps_update_time);
    if (elapsed_time >= fps_update_interval) {
      fps = frame_count / elapsed_time;
      frame_count = 0;
      last_fps_update_time = frame_end;
    }

// ==================== 简化可视化显示 ====================
Eigen::Vector3d euler_deg = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3;

int text_y = 30;
int line_height = 25;

// 1. 显示帧率和状态
cv::Scalar fps_color = (fps > 30) ? cv::Scalar(0, 255, 0) : 
                      (fps > 15) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
tools::draw_text(img, fmt::format("FPS: {:.1f} | {}", fps, planner_status), 
                {10, text_y}, fps_color);
text_y += line_height;

// 2. 显示云台姿态
tools::draw_text(img, 
                fmt::format("Gimbal: Yaw{:.1f}deg Pitch{:.1f}deg", 
                           euler_deg[0], euler_deg[2]),
                {10, text_y}, {255, 255, 255});
text_y += line_height;

// 3. 显示控制命令角度
cv::Scalar cmd_color = command.control ? cv::Scalar(154, 50, 205) : cv::Scalar(100, 100, 100);
std::string cmd_text = command.control ? 
    fmt::format("Cmd: Yaw{:.1f}deg Pitch{:.1f}deg", 
               command.yaw * 57.3, command.pitch * 57.3) :
    "Cmd: INACTIVE";
tools::draw_text(img, cmd_text, {10, text_y}, cmd_color);
text_y += line_height;

// 4. 显示射击状态
cv::Scalar shoot_color;
std::string shoot_text;

if (command.shoot) {
    shoot_color = cv::Scalar(0, 0, 255);
    shoot_text = "FIRING";
} else if (current_plan.control) {
    shoot_color = cv::Scalar(0, 255, 0);
    shoot_text = "AIMING";
} else {
    shoot_color = cv::Scalar(200, 200, 200);
    shoot_text = "STANDBY";
}

tools::draw_text(img, shoot_text, {10, text_y}, shoot_color);
text_y += line_height;

// ==================== 目标可视化 ====================
if (current_target.has_value()) {
  auto target = current_target.value();
  
  // 绘制所有装甲板位置（绿色）
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  for (const Eigen::Vector4d & xyza : armor_xyza_list) {
    auto image_points = solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
    tools::draw_points(img, image_points, {0, 255, 0});
  }

  // 显示瞄准点
  if (planner_initialized) {
    Eigen::Vector4d aim_xyza = planner.debug_xyza;
    auto image_points = solver.reproject_armor(
      aim_xyza.head<3>(), aim_xyza[3], target.armor_type, target.name);
    
    // 简化瞄准点颜色
    cv::Scalar point_color = command.shoot ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    tools::draw_points(img, image_points, point_color, 4);
  }
}

// 预测状态提示
if (planner_status.find("PREDICTING") != std::string::npos) {
  tools::draw_text(img, "PREDICTING...", {10, text_y}, {255, 255, 0});
  text_y += line_height;
}

// -------------------- 观测与绘图（简化） --------------------
nlohmann::json data;
data["gimbal_yaw"] = euler_deg[0];
data["cmd_yaw"] = command.yaw * 57.3;
data["shoot"] = command.shoot;
data["gimbal_pitch"] = euler_deg[2];
data["cmd_pitch"] = command.pitch * 57.3;
data["planner_state"] = planner_status;
data["control"] = command.control;
data["fps"] = fps;

plotter.plot(data);

cv::resize(img, img, {}, 0.5, 0.5);
cv::imshow("Planner Auto Aim", img);
    if (cv::waitKey(1) == 'q') break;
  }
  
  return 0;
}