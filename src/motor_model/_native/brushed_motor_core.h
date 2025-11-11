#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <vector>

namespace motor_model::native {

constexpr double rpm_per_volt_to_rad_per_sec_per_volt_factor =
    2.0 * 3.14159265358979323846 / 60.0;
constexpr double rad_per_sec_per_volt_to_rpm_per_volt_factor =
    60.0 / (2.0 * 3.14159265358979323846);

double rpm_per_volt_to_rad_per_sec_per_volt(double value);
double rad_per_sec_per_volt_to_rpm_per_volt(double value);

struct SimulationResult {
  std::vector<double> time;
  std::vector<double> current;
  std::vector<double> speed;
  std::vector<double> position;
  std::vector<double> torque;
  std::vector<double> voltage;
  std::vector<double> lvdt_time;
  std::vector<double> lvdt;
};

class FeedbackController {
 public:
  virtual ~FeedbackController() = default;
  virtual void reset(double initial_measurement, double initial_current, double initial_speed) = 0;
  virtual double update(double time, double measurement) = 0;
};

using VoltageSource = std::function<double(double)>;

class BrushedMotorModel {
 public:
  BrushedMotorModel(double resistance = 28.0,
                    double inductance = 16e-3,
                    double kv = rpm_per_volt_to_rad_per_sec_per_volt(7.0),
                    double inertia = 4.8e-4,
                    double viscous_friction = 1.9e-4,
                    double coulomb_friction = 2.1e-2,
                    double static_friction = 2.4e-2,
                    double stop_speed_threshold = 1e-4,
                    double spring_constant = 9.5e-4,
                    double spring_compression_ratio = 0.4,
                    double lvdt_full_scale = 0.1,
                    double lvdt_noise_std = 5e-3,
                    int integration_substeps = 1,
                    std::optional<std::uint32_t> rng_seed = std::nullopt);

  double speed_constant_rpm_per_volt() const;
  int integration_substeps() const { return integration_substeps_; }

  SimulationResult simulate(VoltageSource voltage,
                            double duration,
                            double dt,
                            double initial_speed = 0.0,
                            double initial_current = 0.0,
                            VoltageSource load_torque = constant_voltage_source(0.0),
                            std::optional<double> measurement_period = std::nullopt,
                            FeedbackController *controller = nullptr,
                            std::optional<double> controller_period = std::nullopt) const;

  static VoltageSource constant_voltage_source(double value);

 private:
  double spring_torque(double position) const;
  double lvdt_measurement(double position) const;
  static double sign(double value);

  double resistance_;
  double inductance_;
  double kv_;
  double inertia_;
  double viscous_friction_;
  double coulomb_friction_;
  double static_friction_;
  double stop_speed_threshold_;
  double spring_constant_;
  double spring_compression_ratio_;
  double lvdt_full_scale_;
  double lvdt_noise_std_;
  int integration_substeps_;
  double ke_;
  double kt_;
  mutable std::mt19937 rng_;
  mutable std::normal_distribution<double> normal_dist_;
};

}  // namespace motor_model::native
