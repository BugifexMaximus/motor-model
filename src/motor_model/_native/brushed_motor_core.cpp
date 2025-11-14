#include "brushed_motor_core.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace motor_model::native {

namespace {
double round_to_int(double value) {
  return std::round(value);
}

int checked_steps(double duration, double dt) {
  if (dt <= 0.0) {
    throw std::invalid_argument("dt must be positive");
  }
  if (duration <= 0.0) {
    throw std::invalid_argument("duration must be positive");
  }
  const double steps_real = duration / dt;
  const double steps_rounded = round_to_int(steps_real);
  if (!std::isfinite(steps_real)) {
    throw std::invalid_argument("duration/dt must be finite");
  }
  if (std::abs(steps_real - steps_rounded) > 1e-9) {
    throw std::invalid_argument("duration must be an integer multiple of dt");
  }
  return static_cast<int>(steps_rounded);
}

int checked_multiple(double value, double dt, const char *name) {
  const double steps_real = value / dt;
  const double steps_rounded = round_to_int(steps_real);
  if (!std::isfinite(steps_real) || steps_rounded <= 0.0) {
    throw std::invalid_argument(std::string(name) + " must be a positive multiple of dt");
  }
  if (std::abs(steps_real - steps_rounded) > 1e-9) {
    throw std::invalid_argument(std::string(name) + " must be a multiple of dt");
  }
  return static_cast<int>(steps_rounded);
}

}  // namespace

double rpm_per_volt_to_rad_per_sec_per_volt(double value) {
  return value * rpm_per_volt_to_rad_per_sec_per_volt_factor;
}

double rad_per_sec_per_volt_to_rpm_per_volt(double value) {
  return value * rad_per_sec_per_volt_to_rpm_per_volt_factor;
}

BrushedMotorModel::BrushedMotorModel(double resistance,
                                     double inductance,
                                     double kv,
                                     double inertia,
                                     double viscous_friction,
                                     double coulomb_friction,
                                     double static_friction,
                                     double stop_speed_threshold,
                                     double spring_constant,
                                     double spring_compression_ratio,
                                     double spring_zero_position,
                                     double lvdt_full_scale,
                                     double lvdt_noise_std,
                                     int integration_substeps,
                                     std::optional<std::uint32_t> rng_seed)
    : resistance_(resistance),
      inductance_(inductance),
      kv_(kv),
      inertia_(inertia),
      viscous_friction_(viscous_friction),
      coulomb_friction_(coulomb_friction),
      static_friction_(static_friction),
      stop_speed_threshold_(stop_speed_threshold),
      spring_constant_(spring_constant),
      spring_compression_ratio_(spring_compression_ratio),
      spring_zero_position_(spring_zero_position),
      lvdt_full_scale_(lvdt_full_scale),
      lvdt_noise_std_(lvdt_noise_std),
      integration_substeps_(integration_substeps),
      ke_(1.0 / kv),
      kt_(1.0 / kv),
      rng_(rng_seed.value_or(std::random_device{}())),
      normal_dist_(0.0, lvdt_noise_std_) {
  if (kv <= 0.0) {
    throw std::invalid_argument("kv must be positive");
  }
  if (resistance <= 0.0) {
    throw std::invalid_argument("resistance must be positive");
  }
  if (inductance <= 0.0) {
    throw std::invalid_argument("inductance must be positive");
  }
  if (inertia <= 0.0) {
    throw std::invalid_argument("inertia must be positive");
  }
  if (spring_constant < 0.0) {
    throw std::invalid_argument("spring_constant must be non-negative");
  }
  if (spring_compression_ratio < 0.0 || spring_compression_ratio > 1.0) {
    throw std::invalid_argument("spring_compression_ratio must be between 0 and 1");
  }
  if (!std::isfinite(spring_zero_position)) {
    throw std::invalid_argument("spring_zero_position must be finite");
  }
  if (lvdt_full_scale <= 0.0) {
    throw std::invalid_argument("lvdt_full_scale must be positive");
  }
  if (lvdt_noise_std < 0.0) {
    throw std::invalid_argument("lvdt_noise_std must be non-negative");
  }
  if (integration_substeps <= 0) {
    throw std::invalid_argument("integration_substeps must be positive");
  }
}

void BrushedMotorModel::set_integration_substeps(int value) {
  if (value <= 0) {
    throw std::invalid_argument("integration_substeps must be positive");
  }
  integration_substeps_ = value;
}

double BrushedMotorModel::speed_constant_rpm_per_volt() const {
  return rad_per_sec_per_volt_to_rpm_per_volt(kv_);
}

VoltageSource BrushedMotorModel::constant_voltage_source(double value) {
  return [value](double) { return value; };
}

SimulationResult BrushedMotorModel::simulate(VoltageSource voltage,
                                             double duration,
                                             double dt,
                                             double initial_speed,
                                             double initial_current,
                                             VoltageSource load_torque,
                                             std::optional<double> measurement_period,
                                             FeedbackController *controller,
                                             std::optional<double> controller_period) const {
  const int steps = checked_steps(duration, dt);
  const double measurement_period_value = measurement_period.value_or(dt);
  if (measurement_period_value <= 0.0) {
    throw std::invalid_argument("measurement_period must be positive");
  }
  const int measurement_steps = checked_multiple(measurement_period_value, dt, "measurement_period");

  int controller_steps = 0;
  if (controller != nullptr) {
    const double controller_period_value = controller_period.value_or(measurement_period_value);
    if (controller_period_value <= 0.0) {
      throw std::invalid_argument("controller_period must be positive");
    }
    controller_steps = checked_multiple(controller_period_value, dt, "controller_period");
  } else if (controller_period.has_value()) {
    throw std::invalid_argument("controller_period provided without controller");
  }

  SimulationResult result;
  result.time.reserve(static_cast<std::size_t>(steps) + 1);
  result.current.reserve(static_cast<std::size_t>(steps) + 1);
  result.speed.reserve(static_cast<std::size_t>(steps) + 1);
  result.position.reserve(static_cast<std::size_t>(steps) + 1);
  result.torque.reserve(static_cast<std::size_t>(steps) + 1);
  result.voltage.reserve(static_cast<std::size_t>(steps) + 1);
  result.lvdt_time.reserve(static_cast<std::size_t>(steps) + 1);
  result.lvdt.reserve(static_cast<std::size_t>(steps) + 1);
  result.pi_integrator.reserve(static_cast<std::size_t>(steps) + 1);
  result.model_integrator.reserve(static_cast<std::size_t>(steps) + 1);
  result.planned_voltage.reserve(static_cast<std::size_t>(steps) + 1);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  ControllerDiagnostics last_diag;
  bool have_diag = false;

  auto update_diagnostics = [&](FeedbackController *ctrl) {
    if (ctrl == nullptr) {
      return;
    }
    std::optional<ControllerDiagnostics> diag = ctrl->diagnostics();
    if (diag.has_value()) {
      last_diag = std::move(*diag);
      have_diag = true;
    }
  };

  auto append_diagnostics = [&]() {
    if (!have_diag) {
      result.pi_integrator.push_back(nan);
      result.model_integrator.push_back(nan);
      result.planned_voltage.emplace_back();
      return;
    }

    result.pi_integrator.push_back(last_diag.pi_integrator.value_or(nan));
    result.model_integrator.push_back(last_diag.model_integrator.value_or(nan));
    result.planned_voltage.push_back(last_diag.planned_voltage);
  };

  double current = initial_current;
  double speed = initial_speed;
  double position = 0.0;

  const double initial_measurement = lvdt_measurement(position);

  double voltage_command = 0.0;
  if (controller != nullptr) {
    controller->reset(initial_measurement, current, speed);
    voltage_command = controller->update(0.0, initial_measurement);
    update_diagnostics(controller);
  } else {
    voltage_command = voltage(0.0);
  }

  result.time.push_back(0.0);
  result.current.push_back(current);
  result.speed.push_back(speed);
  result.position.push_back(position);
  result.torque.push_back(kt_ * current);
  result.voltage.push_back(voltage_command);
  result.lvdt_time.push_back(0.0);
  result.lvdt.push_back(initial_measurement);
  append_diagnostics();

  for (int step = 0; step < steps; ++step) {
    const double t = step * dt;
    const double load = load_torque(t);

    const double back_emf = ke_ * speed;
    const double di_dt = (voltage_command - resistance_ * current - back_emf) / inductance_;
    current += di_dt * dt;

    const double electromagnetic_torque = kt_ * current;
    const double spring = spring_torque(position);
    const double available_torque = electromagnetic_torque - load - spring;

    if (std::abs(speed) < stop_speed_threshold_ && std::abs(available_torque) <= static_friction_) {
      speed = 0.0;
    } else {
      double friction_direction = sign(speed);
      if (friction_direction == 0.0) {
        friction_direction = sign(available_torque);
      }
      const double dynamic_friction = coulomb_friction_ * friction_direction + viscous_friction_ * speed;
      const double net_torque = available_torque - dynamic_friction;
      const double angular_accel = net_torque / inertia_;
      speed += angular_accel * dt;
    }

    position += speed * dt;

    const double next_time = (step + 1) * dt;

    result.time.push_back(next_time);
    result.current.push_back(current);
    result.speed.push_back(speed);
    result.position.push_back(position);
    result.torque.push_back(electromagnetic_torque);
    result.voltage.push_back(voltage_command);

    if ((step + 1) % measurement_steps == 0) {
      const double measurement_time = next_time;
      const double measurement = lvdt_measurement(position);
      result.lvdt_time.push_back(measurement_time);
      result.lvdt.push_back(measurement);

      if (controller != nullptr && (step + 1) % controller_steps == 0) {
        voltage_command = controller->update(measurement_time, measurement);
        update_diagnostics(controller);
      }
    }

    append_diagnostics();

    if (controller == nullptr) {
      voltage_command = voltage(next_time);
    }
  }

  return result;
}

double BrushedMotorModel::spring_torque(double position) const {
  if (spring_constant_ == 0.0) {
    return 0.0;
  }
  const double deflection = position - spring_zero_position_;
  if (deflection >= 0.0) {
    return spring_constant_ * deflection;
  }
  return spring_constant_ * spring_compression_ratio_ * deflection;
}

double BrushedMotorModel::lvdt_measurement(double position) const {
  double normalized = position / lvdt_full_scale_;
  if (lvdt_noise_std_ > 0.0) {
    if (normal_dist_.stddev() != lvdt_noise_std_) {
      normal_dist_ = std::normal_distribution<double>(0.0, lvdt_noise_std_);
    }
    normalized += normal_dist_(rng_);
  }
  if (normalized > 1.0) {
    return 1.0;
  }
  if (normalized < -1.0) {
    return -1.0;
  }
  return normalized;
}

double BrushedMotorModel::sign(double value) {
  if (value > 0.0) {
    return 1.0;
  }
  if (value < 0.0) {
    return -1.0;
  }
  return 0.0;
}

}  // namespace motor_model::native
