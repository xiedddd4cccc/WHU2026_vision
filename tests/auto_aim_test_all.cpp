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
    io::Camera camera(config_path);  // 工业相机
    auto_aim::YOLO yolo(config_path);
    auto_aim::Solver solver(config_path);
    auto_aim::Tracker tracker(config_path, solver);
    auto_aim::Aimer aimer(config_path);
    io::Gimbal gimbal(config_path);  // 串口通信对象

    cv::Mat img;

    double fps = 0.0;
    auto last_frame_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    constexpr double fps_update_interval = 1.0;

    // -------------------- 射击控制相关变量 --------------------
    io::Command last_command;  // 保存上一帧命令
    bool shoot_enabled = false;  // 射击使能状态
    constexpr double MAX_SHOOT_ANGLE_DIFF = 2.0;  // 最大允许射击角度差(度)
    constexpr int MIN_CONSECUTIVE_FRAMES = 3;  // 最小连续稳定帧数
    int stable_frame_count = 0;  // 连续稳定帧数计数

    // -------------------- 主循环 --------------------
    while (!exiter.exit()) {
        auto frame_start = std::chrono::steady_clock::now();

        camera.read(img, frame_start);  // 读取图像和时间戳
        if (img.empty()) break;

        // -------------------- 云台状态 --------------------
        Eigen::Quaterniond gimbal_q = gimbal.q(frame_start);
        solver.set_R_gimbal2world(gimbal_q);

        // -------------------- 自瞄核心 --------------------
        auto armors = yolo.detect(img, 0);
        auto targets = tracker.track(armors, frame_start);
        auto command = aimer.aim(targets, frame_start, 12, false);

        // -------------------- 射击控制逻辑 --------------------
        bool should_shoot = false;
        
        // 射击条件判断
        if (!targets.empty() && 
            aimer.debug_aim_point.valid && 
            command.control) {
            
            // 计算当前帧与上一帧的角度变化
            double yaw_diff = std::abs(command.yaw - last_command.yaw) * 57.3;  // 弧度转度
            
            if (yaw_diff < MAX_SHOOT_ANGLE_DIFF) {
                stable_frame_count++;
                // 需要连续多帧稳定才允许射击
                if (stable_frame_count >= MIN_CONSECUTIVE_FRAMES) {
                    should_shoot = true;
                    shoot_enabled = true;
                }
            } else {
                stable_frame_count = 0;
                shoot_enabled = false;
            }
        } else {
            stable_frame_count = 0;
            shoot_enabled = false;
        }

        // 更新射击命令
        command.shoot = should_shoot;
        
        // 保存当前命令供下一帧比较
        if (command.control) {
            last_command = command;
        }

        // -------------------- 串口发送控制命令 --------------------
        gimbal.send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);

        // -------------------- 帧率计算 --------------------
        auto frame_end = std::chrono::steady_clock::now();
        frame_count++;
        double elapsed_time = tools::delta_time(frame_end, last_frame_time);

        if (elapsed_time >= fps_update_interval) {
            fps = frame_count / elapsed_time;
            frame_count = 0;
            last_frame_time = frame_end;
            tools::logger()->info(
                "Current FPS: {:.1f}, Tracker State: {}, Stable Frames: {}, Shoot: {}", 
                fps, tracker.state(), stable_frame_count, should_shoot);
        }

        // -------------------- 可视化 --------------------
        Eigen::Vector3d euler = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3;

        cv::Scalar state_color;
        std::string tracker_state = tracker.state();
        if (tracker_state == "tracking") state_color = {0, 255, 0};
        else if (tracker_state == "temp_lost") state_color = {0, 255, 255};
        else if (tracker_state == "detecting") state_color = {255, 255, 0};
        else state_color = {0, 0, 255};

        // 射击状态显示颜色
        cv::Scalar shoot_color = should_shoot ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255);

        tools::draw_text(img, fmt::format("State: {}", tracker_state), {10, 30}, state_color);
        tools::draw_text(img, fmt::format("gimbal yaw:{:.2f}, pitch:{:.2f}", euler[0], euler[2]), {10, 60}, {255, 255, 255});
        tools::draw_text(img, fmt::format("cmd yaw:{:.2f}, pitch:{:.2f}", command.yaw * 57.3, command.pitch * 57.3), {10, 90}, {154, 50, 205});
        tools::draw_text(img, fmt::format("SHOOT: {}, Stable: {}/{}", should_shoot, stable_frame_count, MIN_CONSECUTIVE_FRAMES), {10, 120}, shoot_color);

        // 显示角度变化信息
        double yaw_diff_deg = std::abs(command.yaw - last_command.yaw) * 57.3;
        tools::draw_text(img, fmt::format("Yaw Diff: {:.2f}° (Max: {}°)", yaw_diff_deg, MAX_SHOOT_ANGLE_DIFF), {10, 150}, 
                        yaw_diff_deg < MAX_SHOOT_ANGLE_DIFF ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255));

        if (!targets.empty() && aimer.debug_aim_point.valid) {
            auto &aim_point = aimer.debug_aim_point;
            Eigen::Vector4d aim_xyza;
            for (int i = 0; i < 4; ++i) aim_xyza[i] = static_cast<double>(aim_point.xyza[i]);

            auto &first_target = targets.front();
            auto image_points = solver.reproject_armor(aim_xyza.head<3>(), aim_xyza[3], first_target.armor_type, first_target.name);
            
            // 根据射击状态改变绘制颜色
            cv::Scalar reproject_color = should_shoot ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            tools::draw_points(img, image_points, reproject_color);
        }

        // -------------------- 绘图 --------------------
        nlohmann::json data;
        data["gimbal_yaw"] = euler[0];
        data["cmd_yaw"] = command.yaw * 57.3;
        data["shoot"] = command.shoot;
        data["gimbal_pitch"] = euler[2];
        data["cmd_pitch"] = command.pitch * 57.3;
        data["track_state"] = tracker_state;
        data["control"] = command.control;
        data["stable_frames"] = stable_frame_count;
        data["yaw_diff"] = std::abs(command.yaw - last_command.yaw) * 57.3;

        plotter.plot(data);

        cv::resize(img, img, {}, 0.5, 0.5);
        cv::imshow("reprojection", img);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}