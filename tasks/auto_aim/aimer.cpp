#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{
  Aimer::Aimer(const std::string &config_path)
      : left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
  {
    auto yaml = YAML::LoadFile(config_path);
    yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;     // degree to rad
    pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3; // degree to rad
    high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
    low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
    decision_speed_ = yaml["decision_speed"].as<double>();
    if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined())
    {
      left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() / 57.3;   // degree to rad
      right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() / 57.3; // degree to rad
      tools::logger()->info("[Aimer] successfully loading shootmode");
    }
  }

  io::Command Aimer::aim(
      std::list<Target> targets,
      std::chrono::steady_clock::time_point timestamp,
      double bullet_speed,
      bool to_now,
      Eigen::Vector3d ypr)
  {
    if (targets.empty())
    {
      return {false, false, 0.0, 0.0};
    }

    // 校正子弹速度
    if (bullet_speed < 10.0 || bullet_speed > 30.0)
    {
      bullet_speed = 15.0;
    }

    auto target = targets.front();
    const double w = std::abs(target.ekf_x()[7]); // 角速度 rad/s

    // 延迟时间：根据转速动态调整（可选）
    const double delay_time = (w > decision_speed_) ? high_speed_delay_time_ : low_speed_delay_time_;

    // 计算开火时刻 t_fire
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      timestamp.time_since_epoch())
                      .count();
    int64_t fire_time_us = static_cast<int64_t>(now_us + delay_time * 1e6);
    auto t_fire = std::chrono::steady_clock::time_point(
        std::chrono::microseconds(fire_time_us));

    // 预测 t_fire 时刻的目标状态
    auto fire_target = target;
    fire_target.predict(t_fire);

    Eigen::Vector3d aim_xyz;
    bool use_armor_point = false;

    // ====== 策略切换阈值 ======
    constexpr double ROTATION_THRESHOLD = 2.0; // rad/s (～115 deg/s)

    if (target.name == ArmorName::outpost)
    {
      // 前哨站一律用前方预测
      aim_xyz = get_outpost_front(fire_target);
      use_armor_point = false;
    }
    else if (w > ROTATION_THRESHOLD)
    {
      // 高速旋转（如英雄/步兵小陀螺）
      aim_xyz = get_top_front(fire_target);
      use_armor_point = false;
    }
    else
    {
      // 低速或静止：选择最优装甲板
      auto aim_point = choose_aim_point(fire_target);
      if (!aim_point.valid)
      {
        tools::logger()->debug("[Aimer] No valid armor point");
        return {false, false, 0.0, 0.0};
      }
      aim_xyz = aim_point.xyza.head(3);
      use_armor_point = true;
    }

    // ====== 弹道解算 ======
    const double dx = aim_xyz.x();
    const double dy = aim_xyz.y();
    const double dz = aim_xyz.z();
    const double d = std::sqrt(dx * dx + dy * dy);

    tools::Trajectory traj(bullet_speed, d, dz);
    if (traj.unsolvable)
    {
      tools::logger()->debug("[Aimer] Trajectory unsolvable");
      return {false, false, 0.0, 0.0};
    }

    double yaw_offset = yaw_offset_;

    double yaw = std::atan2(dy, dx) + yaw_offset;
    double pitch = -(traj.pitch + pitch_offset_);

    yaw = tools::limit_rad(yaw);

    // ====== 开火判断 ======
    bool should_shoot = true;

    if (use_armor_point)
    {
      // 平动/慢转目标：只需云台对准
      const double yaw_err = std::abs(tools::limit_rad(ypr[0] - yaw));
      const double pitch_err = std::abs(ypr[1] - pitch);
      should_shoot = (yaw_err < max_fire_error_yaw_) &&
                     (pitch_err < max_fire_error_pitch_);
    }
    else
    {
      // 高速旋转目标：需确保命中时刻有装甲板在“可打窗口”
      const double flight_time = traj.fly_time;
      auto t_hit = t_fire + std::chrono::microseconds(
                                static_cast<int64_t>(flight_time * 1e6));
      auto hit_target = target;
      hit_target.predict(t_hit);

      const double center_yaw = std::atan2(hit_target.ekf_x()[2], hit_target.ekf_x()[0]);
      const double angular_vel = hit_target.ekf_x()[7];
      const int sig = (angular_vel >= 0) ? +1 : -1;

      // 可打窗口大小（快转放宽）
      const double fire_window = (std::abs(angular_vel) > 5.0) ? 0.04 : 0.02; // rad

      should_shoot = false;
      for (const auto &armor : hit_target.armor_xyza_list())
      {
        const double armor_yaw = armor[3];
        const double delta = tools::limit_rad(center_yaw - armor_yaw);
        if ((sig * delta) >= 0 && (sig * delta) <= fire_window)
        {
          should_shoot = true;
          break;
        }
      }

      // 还需云台对准
      if (should_shoot)
      {
        const double yaw_err = std::abs(tools::limit_rad(ypr[0] - yaw));
        const double pitch_err = std::abs(ypr[1] - pitch);
        should_shoot = should_shoot &&
                       (yaw_err < max_fire_error_yaw_) &&
                       (pitch_err < max_fire_error_pitch_);
      }
    }

    // 调试信息（可选）
    debug_aim_point = {
        true, Eigen::Vector4d(aim_xyz.x(), aim_xyz.y(), aim_xyz.z(), 0.0)};

    return {true, should_shoot, yaw, pitch};
  }

  AimPoint Aimer::choose_aim_point(const Target &target)
  {
    Eigen::VectorXd ekf_x = target.ekf_x();
    std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
    auto armor_num = armor_xyza_list.size();
    if (!target.jumped)
      return {true, armor_xyza_list[0]};

    // 整车旋转中心的球坐标yaw
    auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

    std::vector<double> delta_angle_list;
    for (int i = 0; i < armor_num; i++)
    {
      auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
      delta_angle_list.emplace_back(delta_angle);
    }

    // 不考虑小陀螺
    if (std::abs(target.ekf_x()[8]) <= 2 && target.name != ArmorName::outpost)
    {
      // 选择在可射击范围内的装甲板
      std::vector<int> id_list;
      for (int i = 0; i < armor_num; i++)
      {
        if (std::abs(delta_angle_list[i]) > 60 / 57.3)
          continue;
        id_list.push_back(i);
      }
      // 绝无可能
      if (id_list.empty())
      {
        tools::logger()->warn("Empty id list!");
        return {false, armor_xyza_list[0]};
      }

      // 锁定模式：防止在两个都呈45度的装甲板之间来回切换
      if (id_list.size() > 1)
      {
        int id0 = id_list[0], id1 = id_list[1];

        // 未处于锁定模式时，选择delta_angle绝对值较小的装甲板，进入锁定模式
        if (lock_id_ != id0 && lock_id_ != id1)
          lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;

        return {true, armor_xyza_list[lock_id_]};
      }

      // 只有一个装甲板在可射击范围内时，退出锁定模式
      lock_id_ = -1;
      return {true, armor_xyza_list[id_list[0]]};
    }

    double coming_angle, leaving_angle;
    if (target.name == ArmorName::outpost)
    {
      coming_angle = 70 / 57.3;
      leaving_angle = 30 / 57.3;
    }
    else
    {
      coming_angle = comming_angle_;
      leaving_angle = leaving_angle_;
    }

    // 在小陀螺时，一侧的装甲板不断出现，另一侧的装甲板不断消失，显然前者被打中的概率更高
    for (int i = 0; i < armor_num; i++)
    {
      if (std::abs(delta_angle_list[i]) > coming_angle)
        continue;
      if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle)
        return {true, armor_xyza_list[i]};
      if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle)
        return {true, armor_xyza_list[i]};
    }

    return {false, armor_xyza_list[0]};
  }

Eigen::Vector3d Aimer::get_outpost_front(const Target & target)
{
  const auto ekf_x = target.ekf_x();

  Eigen::Vector3d center(ekf_x[0], ekf_x[2], ekf_x[4]);
  Eigen::Vector2d center_xy = center.head<2>();

  double d_center = center_xy.norm();
  if (d_center < 1e-3)
    return center;

  double r = (ekf_x.size() > 8) ? ekf_x[8] : 0.15;
  double horizontal_distance = d_center - r;
  Eigen::Vector2d hit_xy =
      (horizontal_distance / d_center) * center_xy;

  const auto armors = target.armor_xyza_list();
  if (armors.empty())
    return {hit_xy[0], hit_xy[1], ekf_x[4]};

  int best_id = 0;
  double min_d2 = 1e9;

  for (size_t i = 0; i < armors.size(); ++i) {
    double d2 = armors[i][0] * armors[i][0] +
                armors[i][1] * armors[i][1];
    if (d2 < min_d2) {
      min_d2 = d2;
      best_id = i;
    }
  }

  double z_real = armors[best_id][2];

  return {hit_xy[0], hit_xy[1], z_real};
}

  Eigen::Vector3d Aimer::get_top_front(const Target &target)
  {
    auto ekf_x = target.ekf_x();
    Eigen::Vector3d center = {ekf_x[0], ekf_x[2], ekf_x[4]};
    Eigen::Vector2d center_xy = center.head(2);
    double r = ekf_x.size() > 8 ? ekf_x[8] : 0.15;
    double d_center = center_xy.norm();
    if (d_center < 1e-3)
      return center;

    double horizonal_distance = d_center - r;
    Eigen::Vector2d hit_point_xy = (horizonal_distance / d_center) * center_xy;

    auto armors = target.armor_xyza_list();
    if (armors.empty())
    {
      return {hit_point_xy[0], hit_point_xy[1], ekf_x[4]};
    }

    double w = ekf_x[7]; // 角速度
    if (std::abs(w) < 1e-3)
    {
      // 不转，返回最近装甲板高度
      int min_id = 0;
      double min_d2 = 1e9;
      for (size_t i = 0; i < armors.size(); ++i)
      {
        double d2 = armors[i][0] * armors[i][0] + armors[i][1] * armors[i][1];
        if (d2 < min_d2)
        {
          min_d2 = d2;
          min_id = i;
        }
      }
      return {hit_point_xy[0], hit_point_xy[1], armors[min_id][2]};
    }

    double T = 2 * M_PI / std::abs(w);
    double center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
    int best_id = 0;
    double min_dt = 1e9;

    for (size_t i = 0; i < armors.size(); ++i)
    {
      double armor_yaw = armors[i][3];
      double delta_yaw = tools::limit_rad(armor_yaw - center_yaw); // [-π, π]

      // 计算从当前到该装甲板转到正面所需时间
      double dt;
      if (w > 0)
      {
        // 逆时针转：正面是 yaw=0 方向，装甲板需从负角度转过来
        dt = (delta_yaw <= 0) ? -delta_yaw / w : (2 * M_PI - delta_yaw) / w;
      }
      else
      {
        // 顺时针转
        dt = (delta_yaw >= 0) ? delta_yaw / (-w) : (2 * M_PI + delta_yaw) / (-w);
      }

      if (dt < min_dt)
      {
        min_dt = dt;
        best_id = i;
      }
    }

    return {hit_point_xy[0], hit_point_xy[1], armors[best_id][2]};
  }
} // namespace auto_aim