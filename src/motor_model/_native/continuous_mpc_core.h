#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace motor_model {

struct MotorParams {
    double resistance{28.0};
    double inductance{16e-3};
    double kv{0.0};
    double inertia{4.8e-4};
    double viscous_friction{1.9e-4};
    double coulomb_friction{2.1e-2};
    double static_friction{2.4e-2};
    double stop_speed_threshold{1e-4};
    double spring_constant{9.5e-4};
    double spring_compression_ratio{0.4};
    double spring_zero_position{0.0};
    double lvdt_full_scale{0.1};
    double ke{0.0};
    double kt{0.0};
    int integration_substeps{1};

    MotorParams() = default;

    MotorParams(double resistance,
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
                double ke,
                double kt,
                int integration_substeps)
        : resistance(resistance),
          inductance(inductance),
          kv(kv),
          inertia(inertia),
          viscous_friction(viscous_friction),
          coulomb_friction(coulomb_friction),
          static_friction(static_friction),
          stop_speed_threshold(stop_speed_threshold),
          spring_constant(spring_constant),
          spring_compression_ratio(spring_compression_ratio),
          spring_zero_position(spring_zero_position),
          lvdt_full_scale(lvdt_full_scale),
          ke(ke),
          kt(kt),
          integration_substeps(integration_substeps) {}

    MotorParams WithInductance(double new_inductance) const {
        MotorParams copy = *this;
        copy.inductance = std::max(new_inductance, 1e-9);
        return copy;
    }

    double SpringTorque(double position) const {
        if (spring_constant == 0.0) {
            return 0.0;
        }
        double deflection = position - spring_zero_position;
        if (deflection >= 0.0) {
            return spring_constant * deflection;
        }
        return spring_constant * spring_compression_ratio * deflection;
    }

    static double Sign(double value) {
        if (value > 0.0) {
            return 1.0;
        }
        if (value < 0.0) {
            return -1.0;
        }
        return 0.0;
    }
};

struct MPCWeightsValues {
    double position{300.0};
    double speed{0.5};
    double voltage{0.02};
    double delta_voltage{0.25};
    double terminal_position{700.0};
};

inline double ClampSymmetric(double value, double limit) {
    return std::max(-limit, std::min(limit, value));
}

constexpr double kDutyCycleSupplyVoltage = 28.0;

inline double MeasurementToPosition(double measurement, const MotorParams& motor) {
    double clamped = std::max(-1.0, std::min(1.0, measurement));
    return clamped * motor.lvdt_full_scale;
}

inline double PositionToMeasurement(double position, const MotorParams& motor) {
    double normalized = position / motor.lvdt_full_scale;
    return std::max(-1.0, std::min(1.0, normalized));
}

inline bool IsClose(double a, double b, double rel_tol, double abs_tol) {
    return std::abs(a - b) <= std::max(rel_tol * std::max(std::abs(a), std::abs(b)), abs_tol);
}

class ContMPCController {
   public:
    ContMPCController(
        MotorParams motor,
        double dt,
        int horizon,
        double voltage_limit,
        double target_lvdt,
        MPCWeightsValues weights = MPCWeightsValues{},
        double position_tolerance = 0.02,
        double static_friction_penalty = 50.0,
        std::optional<double> friction_compensation = std::nullopt,
        double auto_fc_gain = 2.5,
        double auto_fc_floor = 0.0,
        std::optional<double> auto_fc_cap = std::nullopt,
        double friction_blend_error_low = 0.2,
        double friction_blend_error_high = 0.5,
        int internal_substeps = 30,
        bool robust_electrical = true,
        std::optional<double> electrical_alpha = std::nullopt,
        double inductance_rel_uncertainty = 0.5,
        double pd_blend = 0.7,
        double pd_kp = 6.0,
        double pd_kd = 0.4,
        double pi_ki = 0.0,
        double pi_limit = 5.0,
        bool pi_gate_saturation = true,
        bool pi_gate_blocked = true,
        bool pi_gate_error_band = false,
        bool pi_leak_near_setpoint = true,
        bool use_model_integrator = false,
        int opt_iters = 10,
        double opt_step = 0.1,
        std::optional<double> opt_eps = std::nullopt);

    // Public attributes mirroring the Python implementation.
    double dt;
    int horizon;
    double voltage_limit;
    double target_lvdt;
    double position_tolerance;
    double static_friction_penalty;
    int internal_substeps;
    bool robust_electrical;
    double inductance_rel_uncertainty;
    double pd_blend;
    double pd_kp;
    double pd_kd;
    double pi_ki;
    double pi_limit;
    bool pi_gate_saturation;
    bool pi_gate_blocked;
    bool pi_gate_error_band;
    bool pi_leak_near_setpoint;
    bool use_model_integrator;
    double auto_fc_gain;
    double auto_fc_floor;
    std::optional<double> auto_fc_cap;
    double friction_blend_error_low;
    double friction_blend_error_high;
    double model_bias_limit;
    double friction_compensation;
    bool output_duty_cycle{false};

    int _opt_iters;
    double _opt_step;
    double _opt_eps;
    double _electrical_alpha;
    double _int_err{0.0};
    double _u_bias{0.0};
    std::optional<double> _user_friction_compensation_request;
    double _auto_friction_compensation{0.0};
    std::optional<double> _active_user_friction_compensation;

    std::optional<double> _last_voltage;
    std::optional<double> _last_measured_position;
    std::optional<double> _last_measurement_time;

    const MotorParams& motor_params() const noexcept { return motor_params_; }
    const std::vector<MotorParams>& prediction_models() const noexcept { return prediction_model_params_; }

    const std::array<double, 3>& state() const noexcept { return _state; }
    const std::array<double, 3>& nominal_state() const noexcept { return _nominal_state; }
    const std::vector<double>& u_sequence() const noexcept { return _u_seq; }

    MPCWeightsValues& mutable_weights() noexcept { return weights_; }
    const MPCWeightsValues& weights() const noexcept { return weights_; }
    void set_weights(const MPCWeightsValues& weights) { weights_ = weights; }

    void adapt_to_motor(
        const MotorParams& motor,
        std::optional<double> voltage_limit_arg,
        std::optional<double> friction_compensation_arg);

    void reset(double initial_measurement, double initial_current, double initial_speed);

    double update(double time, double measurement);

   private:
    void Validate_initial_arguments();
    void Reset_internal_sequences();
    void Apply_motor_parameters(
        const MotorParams& motor,
        std::optional<double> friction_compensation_arg);
    std::tuple<double, std::optional<double>, double> Determine_friction_compensation(
        const MotorParams& params,
        std::optional<double> friction_compensation_arg);
    std::vector<MotorParams> Build_prediction_models(const MotorParams& motor);
    std::vector<double> OptimizeSequence(
        const std::array<double, 3>& initial_state,
        double voltage_bias);
    std::tuple<double, std::array<double, 3>, double> EvaluateSequenceSingle(
        const std::array<double, 3>& initial_state,
        const std::vector<double>& sequence,
        const MotorParams& motor,
        double voltage_bias,
        const MPCWeightsValues& weights);
    std::tuple<double, std::array<double, 3>, double> EvaluateSequence(
        const std::array<double, 3>& initial_state,
        const std::vector<double>& sequence,
        double voltage_bias);
    std::array<double, 3> PredictNext(
        const std::array<double, 3>& state,
        double voltage,
        const MotorParams& motor) const;
    double DynamicFrictionCompensation(double position_error) const;

    MotorParams motor_params_;
    std::vector<MotorParams> prediction_model_params_;
    MPCWeightsValues weights_;
    std::array<double, 3> _state{0.0, 0.0, 0.0};
    std::array<double, 3> _nominal_state{0.0, 0.0, 0.0};
    std::vector<double> _u_seq;
};

}  // namespace motor_model

