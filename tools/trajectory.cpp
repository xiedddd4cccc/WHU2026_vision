#include "trajectory.hpp"

#include <cmath>

namespace tools
{

// ================= 物理参数 =================
constexpr double g = 9.7833;
constexpr double rho = 1.169;  // 空气密度 kg/m^3
constexpr double Cd = 0.47;    // 球体阻力系数
constexpr double r = 0.02125;  // 半径 m
constexpr double m = 0.041;    // 质量 kg
constexpr double A = M_PI * r * r;
constexpr double k = 0.5 * rho * Cd * A / m;

// 积分参数
constexpr double dt = 0.001;

// ================= 状态结构 =================
struct State
{
  double x, y;
  double vx, vy;
};

// ================= 加速度 =================
inline void acceleration(double vx, double vy, double & ax, double & ay)
{
  double v = std::sqrt(vx * vx + vy * vy);
  if (v < 1e-6) {
    ax = 0.0;
    ay = -g;
    return;
  }

  double drag = k * v;
  ax = -drag * vx;
  ay = -g - drag * vy;
}

// ================= RK4 单步 =================
State rk4_step(const State & s)
{
  auto deriv = [](const State & s, State & ds) {
    ds.x = s.vx;
    ds.y = s.vy;
    acceleration(s.vx, s.vy, ds.vx, ds.vy);
  };

  State k1, k2, k3, k4, tmp;

  deriv(s, k1);

  tmp = {
    s.x + 0.5 * dt * k1.x, s.y + 0.5 * dt * k1.y, s.vx + 0.5 * dt * k1.vx, s.vy + 0.5 * dt * k1.vy};
  deriv(tmp, k2);

  tmp = {
    s.x + 0.5 * dt * k2.x, s.y + 0.5 * dt * k2.y, s.vx + 0.5 * dt * k2.vx, s.vy + 0.5 * dt * k2.vy};
  deriv(tmp, k3);

  tmp = {s.x + dt * k3.x, s.y + dt * k3.y, s.vx + dt * k3.vx, s.vy + dt * k3.vy};
  deriv(tmp, k4);

  State out;
  out.x = s.x + dt / 6.0 * (k1.x + 2 * k2.x + 2 * k3.x + k4.x);
  out.y = s.y + dt / 6.0 * (k1.y + 2 * k2.y + 2 * k3.y + k4.y);
  out.vx = s.vx + dt / 6.0 * (k1.vx + 2 * k2.vx + 2 * k3.vx + k4.vx);
  out.vy = s.vy + dt / 6.0 * (k1.vy + 2 * k2.vy + 2 * k3.vy + k4.vy);

  return out;
}

// ================= 弹道仿真 =================
double Trajectory::simulate(double pitch, double & time) const
{
  State s;
  s.x = 0.0;
  s.y = 0.0;
  s.vx = v0_ * std::cos(pitch);
  s.vy = v0_ * std::sin(pitch);

  time = 0.0;

  while (s.x < d_) {
    if (s.y < -1.0) break;  // 已明显落地
    s = rk4_step(s);
    time += dt;
    if (time > 5.0) break;  // 防死循环
  }

  return s.y;
}

// ================= 构造函数：反解 pitch =================
Trajectory::Trajectory(double v0, double d, double h) : v0_(v0), d_(d), h_(h)
{
  double low = -0.3;  // -17°
  double high = 1.0;  // 57°
  double mid, y, t;

  bool found = false;

  for (int i = 0; i < 50; ++i) {
    mid = 0.5 * (low + high);
    y = simulate(mid, t);

    if (std::abs(y - h_) < 1e-3) {
      found = true;
      break;
    }

    if (y > h_)
      high = mid;
    else
      low = mid;
  }

  if (!found) {
    unsolvable = true;
    return;
  }

  unsolvable = false;
  pitch = mid;
  simulate(pitch, fly_time);
}

}  // namespace tools