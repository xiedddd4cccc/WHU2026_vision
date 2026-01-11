#pragma once

namespace tools
{

class Trajectory
{
public:
  Trajectory(double v0, double d, double h);

  bool unsolvable = false;
  double pitch = 0.0;     // rad
  double fly_time = 0.0;  // s

private:
  double simulate(double pitch, double & fly_time) const;

  double v0_;
  double d_;
  double h_;
};

}  // namespace tools