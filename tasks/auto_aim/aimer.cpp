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

  if (bullet_speed < 10.0 || bullet_speed > 30.0)
  {
    bullet_speed = 15.0;
  }

  auto target = targets.front();
  double w = std::abs(target.ekf_x()[7]);

  const double delay_time =
      (w > decision_speed_) ? high_speed_delay_time_ : low_speed_delay_time_;

  // ================= 1 计算 fire 时刻 =================
  auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    timestamp.time_since_epoch())
                    .count();

  int64_t fire_time_us = static_cast<int64_t>(now_us + delay_time * 1e6);
  auto t_fire = std::chrono::steady_clock::time_point(std::chrono::microseconds(fire_time_us));

  // ================= 2 预测 fire 时刻 =================
  auto fire_target = target; // 从原始观测开始
  fire_target.predict(t_fire);

  Eigen::Vector3d aim_xyz;
  constexpr double ROTATION_THRESHOLD = 2.0;

  // ---------- 第一次粗算飞行时间 ----------
  Eigen::Vector3d center0 = {
      fire_target.ekf_x()[0],
      fire_target.ekf_x()[2],
      fire_target.ekf_x()[4]};

  double d0 = center0.head<2>().norm();
  tools::Trajectory traj0(bullet_speed, d0, center0.z());

  if (traj0.unsolvable)
  {
    return {false, false, 0.0, 0.0};
  }

  double t_guess = traj0.fly_time;

  // ---------- 预测命中时刻 ----------
  auto t_hit = t_fire + std::chrono::microseconds(static_cast<int64_t>(t_guess * 1e6));

  // 从最原始的 target 开始预测，防止多次 predict 累加产生的平移漂移
  auto hit_target = target;
  hit_target.predict(t_hit);

  // ---------- 用命中时刻状态计算真正瞄准点 ----------
  if (target.name == ArmorName::outpost)
  {
    aim_xyz = get_outpost_front(hit_target);
  }
  else if (w > ROTATION_THRESHOLD)
  {
    aim_xyz = get_top_front(hit_target);
  }
  else
  {
    auto aim_point = choose_aim_point(hit_target);
    if (!aim_point.valid)
    {
      return {false, false, 0.0, 0.0};
    }
    aim_xyz = aim_point.xyza.head(3);
  }

  // ---------- 第二次真正弹道 ----------
  double dx = aim_xyz.x();
  double dy = aim_xyz.y();
  double dz = aim_xyz.z();
  double d = std::sqrt(dx * dx + dy * dy);

  tools::Trajectory traj(bullet_speed, d, dz);
  if (traj.unsolvable)
  {
    return {false, false, 0.0, 0.0};
  }

  // geo_yaw 是子弹实际在空间中飞行的轴线角度
  double geo_yaw = std::atan2(dy, dx);

  double yaw = geo_yaw + yaw_offset_;
  yaw = tools::limit_rad(yaw);
  double pitch = -(traj.pitch + pitch_offset_);

  // ==========================================================
  // 开火判断
  // ==========================================================
  bool should_shoot = true;

  if (w <= ROTATION_THRESHOLD)
  {
    // 平移/低速目标
    double yaw_err = std::abs(tools::limit_rad(ypr[0] - yaw));
    double pitch_err = std::abs(ypr[1] - pitch);

    should_shoot = (yaw_err < max_fire_error_yaw_) && (pitch_err < max_fire_error_pitch_);
  }
  else
  {
    auto check_target = target;
    auto t_hit2 = t_fire + std::chrono::microseconds(static_cast<int64_t>(traj.fly_time * 1e6));
    check_target.predict(t_hit2);

    int sig = (check_target.ekf_x()[7] >= 0) ? +1 : -1;

    double fire_window = (std::abs(check_target.ekf_x()[7]) > 5.0) ? 0.02 : 0.015;

    should_shoot = false;

    // 理想的装甲板法线方向应该是与子弹飞行方向 完全相反
    double ideal_armor_yaw = tools::limit_rad(geo_yaw + M_PI);

    for (auto const &armor : check_target.armor_xyza_list())
    {
      double armor_yaw = armor[3];

      // 判断装甲板法线与理想弹道的偏差
      double delta = tools::limit_rad(ideal_armor_yaw - armor_yaw);

      // sig 控制只在装甲板“迎面转过来”时开火
      // 极窄的 fire_window 保证了击打位置极其靠近装甲板中心
      if ((sig * delta) >= 0 && (sig * delta) <= fire_window)
      {
        should_shoot = true;
        break;
      }
    }

    if (should_shoot)
    {
      // 枪管当前的实际朝向误差也必须满足严苛条件
      double yaw_err = std::abs(tools::limit_rad(ypr[0] - yaw));
      double pitch_err = std::abs(ypr[1] - pitch);

      should_shoot = (yaw_err < max_fire_error_yaw_) && (pitch_err < max_fire_error_pitch_);
    }
  }

  debug_aim_point = {true, Eigen::Vector4d(aim_xyz.x(), aim_xyz.y(), aim_xyz.z(), 0.0)};

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

  // 对于小陀螺的车，我们要打的位置AimPoint
Eigen::Vector3d Aimer::get_top_front(const auto_aim::Target & target_origin)
{
  // 这时候瞄准位置的z坐标是下一块来接紫蛋的装甲板的高度
  auto ekf_x = target_origin.ekf_x();

  Eigen::Vector3d center = {ekf_x[0], ekf_x[2], ekf_x[4]};
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);  // 整车旋转中心的球坐标yaw

  Eigen::Vector2d center_xy = center.head(2);
  double horizonal_distance = center_xy.norm() - ekf_x[8];
  Eigen::Vector2d hit_point_xy = (horizonal_distance / center_xy.norm()) * center_xy;

  auto armors = target_origin.armor_xyza_list();
  int armor_num = armors.size();

  double w = ekf_x[7], T = 2 * M_PI / w;
  int sig_w = w > 0 ? 1 : -1;  // 旋转方向
  int armor_id = -1;
  double dt_min = INFINITY, t;
  if (std::fabs(w) > 3) {
    return {hit_point_xy[0], hit_point_xy[1], ekf_x[4]};
  }
  for (int aim_id = 0; aim_id < armor_num; aim_id++) {
    if (sig_w == 1) {  // w>0
      t = (armors[aim_id][3] > center_yaw) ? T - (armors[aim_id][3] - center_yaw) / w
                                           : (center_yaw - armors[aim_id][3] / w);
    } else {
      t = (armors[aim_id][3] < center_yaw) ? T + (armors[aim_id][3] - center_yaw) / w
                                           : (-center_yaw + armors[aim_id][3] / w);
    }
    if (t < dt_min) {
      armor_id = aim_id;
      dt_min = t;
    }
  }
  return {hit_point_xy[0], hit_point_xy[1], armors[armor_id][2]};
}
} // namespace auto_aim