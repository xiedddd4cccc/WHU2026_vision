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
Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
{
  auto yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;      // degree to rad
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;  // degree to rad
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
  decision_speed_ = yaml["decision_speed"].as<double>();
  if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) {
    left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() / 57.3;    // degree to rad
    right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() / 57.3;  // degree to rad
    tools::logger()->info("[Aimer] successfully loading shootmode");
  }
}

io::Command auto_aim::Aimer::aim(
  std::list<auto_aim::Target> targets, 
  std::chrono::steady_clock::time_point timestamp,
  double bullet_speed, 
  bool to_now,
  Eigen::Vector3d ypr)  // ← 新增参数：当前云台姿态（yaw, pitch, roll）
{
  if (targets.empty()) {
    return {false, false, 0.0, 0.0};
  }

  auto target = targets.front();

  // 校正子弹速度
  if (bullet_speed < 10.0 || bullet_speed > 30.0) {
    bullet_speed = 15.0;
  }

  // 延迟时间
  double delay_time = (std::abs(target.ekf_x()[7]) > decision_speed_) ? high_speed_delay_time_
                                                                      : low_speed_delay_time_;

  // 计算发射时刻 t_fire
  std::chrono::steady_clock::time_point t_fire;
  if (to_now) {
    double dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
    t_fire = timestamp + std::chrono::microseconds(static_cast<int64_t>(dt * 1e6));
  } else {
    t_fire =
      timestamp + std::chrono::microseconds(static_cast<int64_t>((0.005 + delay_time) * 1e6));
  }

  // 预测发射时刻状态
  auto predicted_target = target;
  predicted_target.predict(t_fire);

  Eigen::Vector3d aim_xyz;
  bool should_shoot_based_on_rotation = false;  // 旋转窗口内可打
  bool valid = true;

  const double w = predicted_target.ekf_x()[7];  // 角速度
  const bool is_rotating = std::abs(w) > 1.0;    // 阈值可调

  if (predicted_target.name == ArmorName::outpost && is_rotating) {
    // ==================== 前哨站 ====================
    aim_xyz = get_outpost_front(predicted_target);

    // --- Fire 判断：预测命中时刻 ---
    double d = std::sqrt(aim_xyz.x() * aim_xyz.x() + aim_xyz.y() * aim_xyz.y());
    tools::Trajectory traj(bullet_speed, d, aim_xyz.z());
    if (!traj.unsolvable) {
      double flight_time = traj.fly_time;
      auto t_hit = t_fire + std::chrono::microseconds(static_cast<int64_t>(flight_time * 1e6));

      auto hit_target = target;
      hit_target.predict(t_hit);

      double center_yaw = std::atan2(hit_target.ekf_x()[2], hit_target.ekf_x()[0]);
      double fire_error = 0.01;  // 弧度 ≈ 0.57度

      for (const auto & armor : hit_target.armor_xyza_list()) {
        double delta = tools::limit_rad(center_yaw - armor[3]);  // center - armor_yaw
        if (delta >= 0 && delta <= fire_error) {
          should_shoot_based_on_rotation = true;
          break;
        }
      }
    }

  } else if (is_rotating) {
    // ==================== 小陀螺（top） ====================
    aim_xyz = get_top_front(predicted_target);

    // --- Fire 判断：预测命中时刻 ---
    double d = std::sqrt(aim_xyz.x() * aim_xyz.x() + aim_xyz.y() * aim_xyz.y());
    tools::Trajectory traj(bullet_speed, d, aim_xyz.z());
    if (!traj.unsolvable) {
      double flight_time = traj.fly_time;
      auto t_hit = t_fire + std::chrono::microseconds(static_cast<int64_t>(flight_time * 1e6));

      auto hit_target = target;
      hit_target.predict(t_hit);

      double center_yaw = std::atan2(hit_target.ekf_x()[2], hit_target.ekf_x()[0]);
      double fire_error = (std::abs(w) > 5.0) ? 0.04 : 0.02;  // 快转放宽窗口

      int sig = (w >= 0) ? +1 : -1;
      for (const auto & armor : hit_target.armor_xyza_list()) {
        double delta = tools::limit_rad(center_yaw - armor[3]);
        if ((sig * delta) >= 0 && (sig * delta) <= fire_error) {
          should_shoot_based_on_rotation = true;
          break;
        }
      }
    }

  } else {
    // ==================== 平动目标 ====================
// 第一次预测：发射时刻
    auto target0 = predicted_target; // t_fire 时刻
    auto aim_point0 = choose_aim_point(target0);
    if (!aim_point0.valid) {
        return {false, false, 0.0, 0.0};
    }
    Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);
    double d0 = std::sqrt(xyz0.x() * xyz0.x() + xyz0.y() * xyz0.y());
    tools::Trajectory traj0(bullet_speed, d0, xyz0.z()); 
    if (traj0.unsolvable) {
        return {false, false, 0.0, 0.0};
    }

    // 第二次预测：命中时刻
    auto t_hit = t_fire + std::chrono::microseconds(static_cast<int64_t>(traj0.fly_time * 1e6));
    auto target1 = target;
    target1.predict(t_hit);
    auto aim_point1 = choose_aim_point(target1);
    if (!aim_point1.valid) {
        return {false, false, 0.0, 0.0};
    }
    Eigen::Vector3d xyz1 = aim_point1.xyza.head(3);
    double d1 = std::sqrt(xyz1.x() * xyz1.x() + xyz1.y() * xyz1.y());
    tools::Trajectory traj1(bullet_speed, d1, xyz1.z());
    if (traj1.unsolvable) {
        return {false, false, 0.0, 0.0};
    }

    // 一致性检查（可选，但推荐）
    if (std::abs(traj1.fly_time - traj0.fly_time) > 0.1) {
        // 可打日志，但不直接返回，继续用 traj1
        tools::logger()->debug("[Aimer] Large time error in flat target: {:.3f}", 
                               traj1.fly_time - traj0.fly_time);
    }

    // 使用第二次结果作为最终瞄准点
    double yaw = std::atan2(xyz1.y(), xyz1.x()) + yaw_offset_;
    double pitch = -(traj1.pitch + pitch_offset_); // 注意符号！

    // 判断云台是否对准 → 决定是否开火
    bool is_gimbal_aligned = 
        (std::abs(tools::limit_rad(ypr[0] - yaw)) < max_fire_error_yaw_) &&
        (std::abs(ypr[1] - pitch) < max_fire_error_pitch_);

    debug_aim_point = aim_point1;

    // 平动目标：只要对准就开火（不像旋转目标需等窗口）
    bool should_shoot = is_gimbal_aligned;

    return {true, should_shoot, yaw, pitch};
  }

  // 弹道解算（再次，因为 aim_xyz 已确定）
  double d = std::sqrt(aim_xyz.x() * aim_xyz.x() + aim_xyz.y() * aim_xyz.y());
  tools::Trajectory traj(bullet_speed, d, aim_xyz.z());
  if (traj.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory: v={:.2f}, d={:.2f}, z={:.2f}", bullet_speed, d, aim_xyz.z());
    return {false, false, 0.0, 0.0};
  }

  // 最终角度
  double yaw = std::atan2(aim_xyz.y(), aim_xyz.x()) + yaw_offset_;
  double pitch = -(traj.pitch + pitch_offset_);  // 注意方向（假设 pitch 向下为正）

  // ====== 新增：根据云台当前状态判断是否真正开火 ======
  bool is_gimbal_aligned = 
    (std::abs(tools::limit_rad(ypr[0] - yaw)) < max_fire_error_yaw_) &&
    (std::abs(ypr[1] - pitch) < max_fire_error_pitch_);

  bool should_shoot = should_shoot_based_on_rotation && is_gimbal_aligned;

  debug_aim_point = {
    true, should_shoot, Eigen::Vector4d(aim_xyz.x(), aim_xyz.y(), aim_xyz.z(), 0.0)};

  return {true, should_shoot, yaw, pitch};
}

AimPoint Aimer::choose_aim_point(const Target & target)
{
  const auto & armors = target.armor_xyza_list();
  const int armor_num = armors.size();

  if (armor_num == 0) {
    return {false, false, Eigen::Vector4d::Zero()};
  }

  // ===== 1. 未发生跳变，直接用当前装甲板 =====
  if (!target.jumped) {
    return {true, false, armors[0]};
  }

  const auto ekf_x = target.ekf_x();

  // ===== 2. 整车旋转中心 yaw（与 Shooter 完全一致）=====
  const double center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  // ===== 3. 找 yaw 最接近 center_yaw 的两个装甲板 =====
  int first_id = -1, second_id = -1;
  double min1 = 1e9, min2 = 1e9;

  for (int i = 0; i < armor_num; ++i) {
    double delta = std::abs(
      tools::limit_rad(armors[i][3] - center_yaw));

    if (delta < min1) {
      min2 = min1;
      second_id = first_id;
      min1 = delta;
      first_id = i;
    } else if (delta < min2) {
      min2 = delta;
      second_id = i;
    }
  }

  // ===== 4. 找平面距离最近的两个装甲板 =====
  int dmin_first = -1, dmin_second = -1;
  double dmin1 = 1e9, dmin2 = 1e9;

  for (int i = 0; i < armor_num; ++i) {
    double d2 = armors[i][0] * armors[i][0] +
                armors[i][1] * armors[i][1];

    if (d2 < dmin1) {
      dmin2 = dmin1;
      dmin_second = dmin_first;
      dmin1 = d2;
      dmin_first = i;
    } else if (d2 < dmin2) {
      dmin2 = d2;
      dmin_second = i;
    }
  }

  // ===== 5. yaw / 距离一致性筛选（与 Shooter 行为一致）=====
  auto is_nearest = [&](int id) {
    return id == dmin_first || id == dmin_second;
  };

  if (!is_nearest(first_id)) {
    return {true, false, armors[second_id]};
  }

  if (!is_nearest(second_id)) {
    return {true, false, armors[first_id]};
  }

  // ===== 6. 锁定逻辑（防止两个侧对装甲板来回跳）=====
  const double yaw1 = std::atan2(armors[first_id][1], armors[first_id][0]);
  const double yaw2 = std::atan2(armors[second_id][1], armors[second_id][0]);

  if (lock_id_ != first_id && lock_id_ != second_id) {
    const double err1 =
      std::abs(tools::limit_rad(armors[first_id][3] - yaw1));
    const double err2 =
      std::abs(tools::limit_rad(armors[second_id][3] - yaw2));

    lock_id_ = (err1 < err2) ? first_id : second_id;
  }

  return {true, false, armors[lock_id_]};
}

Eigen::Vector3d Aimer::get_outpost_front(const Target & target)
{
  const auto ekf_x = target.ekf_x();

  // ==================== 1. 旋转中心 ====================
  Eigen::Vector3d center(ekf_x[0], ekf_x[2], ekf_x[4]);
  Eigen::Vector2d center_xy = center.head<2>();

  double d_center = center_xy.norm();
  if (d_center < 1e-3) {
    return center;
  }

  // ==================== 2. 前沿 XY（与 get_top_front 一致） ====================
  double r = (ekf_x.size() > 8) ? ekf_x[8] : 0.15;  // 默认半径 15 cm
  double horizontal_distance = d_center - r;
  Eigen::Vector2d hit_xy = (horizontal_distance / d_center) * center_xy;

  // ==================== 3. 装甲板集合 ====================
  const auto armors = target.armor_xyza_list();
  if (armors.empty()) {
    return {hit_xy[0], hit_xy[1], ekf_x[4]};
  }

  // ==================== 4. 三档高度先验 ====================
  constexpr double dz = 0.1;
  const double z_center = ekf_x[4];
  const std::array<double, 3> z_levels = {
    z_center - dz,
    z_center,
    z_center + dz
  };

  // ==================== 5. 若几乎不转：选最近装甲板高度并吸附 ====================
  double w = ekf_x[7];
  if (std::abs(w) < 1e-3) {
    int best_id = 0;
    double min_d2 = 1e9;
    for (size_t i = 0; i < armors.size(); ++i) {
      double d2 = armors[i][0]*armors[i][0] + armors[i][1]*armors[i][1];
      if (d2 < min_d2) {
        min_d2 = d2;
        best_id = i;
      }
    }

    // 高度吸附
    double z_obs = armors[best_id][2];
    double best_z = z_levels[0];
    double min_err = std::abs(z_obs - best_z);
    for (double z : z_levels) {
      double err = std::abs(z_obs - z);
      if (err < min_err) {
        min_err = err;
        best_z = z;
      }
    }

    return {hit_xy[0], hit_xy[1], best_z};
  }

  // ==================== 6. 旋转：选“最先转到正面”的装甲板 ====================
  double center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  int best_id = 0;
  double min_dt = 1e9;

  for (size_t i = 0; i < armors.size(); ++i) {
    double armor_yaw = armors[i][3];
    double delta_yaw = tools::limit_rad(armor_yaw - center_yaw);

    double dt;
    if (w > 0) {
      dt = (delta_yaw <= 0) ? -delta_yaw / w : (2 * M_PI - delta_yaw) / w;
    } else {
      dt = (delta_yaw >= 0) ? delta_yaw / (-w) : (2 * M_PI + delta_yaw) / (-w);
    }

    if (dt < min_dt) {
      min_dt = dt;
      best_id = i;
    }
  }

  // ==================== 7. 高度吸附到三档 ====================
  double z_obs = armors[best_id][2];
  double best_z = z_levels[0];
  double min_err = std::abs(z_obs - best_z);
  for (double z : z_levels) {
    double err = std::abs(z_obs - z);
    if (err < min_err) {
      min_err = err;
      best_z = z;
    }
  }

  return {hit_xy[0], hit_xy[1], best_z};
}


Eigen::Vector3d Aimer::get_top_front(const Target & target)
{
  auto ekf_x = target.ekf_x();
  Eigen::Vector3d center = {ekf_x[0], ekf_x[2], ekf_x[4]};
  Eigen::Vector2d center_xy = center.head(2);
  double r = ekf_x.size() > 8 ? ekf_x[8] : 0.15;
  double d_center = center_xy.norm();
  if (d_center < 1e-3) return center;

  double horizonal_distance = d_center - r;
  Eigen::Vector2d hit_point_xy = (horizonal_distance / d_center) * center_xy;

  auto armors = target.armor_xyza_list();
  if (armors.empty()) {
    return {hit_point_xy[0], hit_point_xy[1], ekf_x[4]};
  }

  double w = ekf_x[7];  // 角速度
  if (std::abs(w) < 1e-3) {
    // 不转，返回最近装甲板高度
    int min_id = 0;
    double min_d2 = 1e9;
    for (size_t i = 0; i < armors.size(); ++i) {
      double d2 = armors[i][0] * armors[i][0] + armors[i][1] * armors[i][1];
      if (d2 < min_d2) {
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

  for (size_t i = 0; i < armors.size(); ++i) {
    double armor_yaw = armors[i][3];
    double delta_yaw = tools::limit_rad(armor_yaw - center_yaw);  // [-π, π]

    // 计算从当前到该装甲板转到正面所需时间
    double dt;
    if (w > 0) {
      // 逆时针转：正面是 yaw=0 方向，装甲板需从负角度转过来
      dt = (delta_yaw <= 0) ? -delta_yaw / w : (2 * M_PI - delta_yaw) / w;
    } else {
      // 顺时针转
      dt = (delta_yaw >= 0) ? delta_yaw / (-w) : (2 * M_PI + delta_yaw) / (-w);
    }

    if (dt < min_dt) {
      min_dt = dt;
      best_id = i;
    }
  }

  return {hit_point_xy[0], hit_point_xy[1], armors[best_id][2]};
}
}  // namespace auto_aim