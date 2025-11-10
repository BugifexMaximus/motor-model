// Copyright (c) 2024
//
// Native pybind11 implementation of the continuous-action MPC controller.

#pragma once

#include <pybind11/attr.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace motor_model {

namespace {

struct MotorParams {
    double resistance;
    double inductance;
    double kv;
    double inertia;
    double viscous_friction;
    double coulomb_friction;
    double static_friction;
    double stop_speed_threshold;
    double spring_constant;
    double spring_compression_ratio;
    double lvdt_full_scale;
    double ke;
    double kt;
    int integration_substeps;

    static MotorParams FromPython(const py::object& motor) {
        MotorParams params;
        params.resistance = motor.attr("resistance").cast<double>();
        params.inductance = motor.attr("inductance").cast<double>();
        params.kv = motor.attr("kv").cast<double>();
        params.inertia = motor.attr("inertia").cast<double>();
        params.viscous_friction = motor.attr("viscous_friction").cast<double>();
        params.coulomb_friction = motor.attr("coulomb_friction").cast<double>();
        params.static_friction = motor.attr("static_friction").cast<double>();
        params.stop_speed_threshold = motor.attr("stop_speed_threshold").cast<double>();
        params.spring_constant = motor.attr("spring_constant").cast<double>();
        params.spring_compression_ratio = motor.attr("spring_compression_ratio").cast<double>();
        params.lvdt_full_scale = motor.attr("lvdt_full_scale").cast<double>();
        params.ke = motor.attr("_ke").cast<double>();
        params.kt = motor.attr("_kt").cast<double>();
        params.integration_substeps = motor.attr("integration_substeps").cast<int>();
        return params;
    }

    MotorParams WithInductance(double new_inductance) const {
        MotorParams copy = *this;
        copy.inductance = std::max(new_inductance, 1e-9);
        return copy;
    }

    double SpringTorque(double position) const {
        if (spring_constant == 0.0) {
            return 0.0;
        }
        if (position >= 0.0) {
            return spring_constant * position;
        }
        return spring_constant * spring_compression_ratio * position;
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
    double position;
    double speed;
    double voltage;
    double delta_voltage;
    double terminal_position;
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

inline MPCWeightsValues ParseWeights(const py::object& weights_obj) {
    MPCWeightsValues values;
    values.position = weights_obj.attr("position").cast<double>();
    values.speed = weights_obj.attr("speed").cast<double>();
    values.voltage = weights_obj.attr("voltage").cast<double>();
    values.delta_voltage = weights_obj.attr("delta_voltage").cast<double>();
    values.terminal_position = weights_obj.attr("terminal_position").cast<double>();
    return values;
}

inline py::object DefaultWeightsObject() {
    static py::object weights_type =
        py::module::import("motor_model._mpc_common").attr("MPCWeights");
    return weights_type();
}

inline py::object CloneMotorWithInductance(const py::object& motor, double inductance) {
    static py::object clone_helper =
        py::module::import("motor_model._mpc_common").attr("_clone_motor_with_inductance");
    return clone_helper(motor, py::arg("inductance") = inductance);
}

}  // namespace

class ContMPCController {
   public:
    ContMPCController(
        py::object motor,
        double dt,
        int horizon,
        double voltage_limit,
        double target_lvdt,
        py::object weights,
        double position_tolerance,
        double static_friction_penalty,
        std::optional<double> friction_compensation,
        double auto_fc_gain,
        double auto_fc_floor,
        std::optional<double> auto_fc_cap,
        double friction_blend_error_low,
        double friction_blend_error_high,
        int internal_substeps,
        bool robust_electrical,
        std::optional<double> electrical_alpha,
        double inductance_rel_uncertainty,
        double pd_blend,
        double pd_kp,
        double pd_kd,
        double pi_ki,
        double pi_limit,
        bool pi_gate_saturation,
        bool pi_gate_blocked,
        bool pi_gate_error_band,
        bool pi_leak_near_setpoint,
        bool use_model_integrator,
        int opt_iters,
        double opt_step,
        std::optional<double> opt_eps
    )
        : dt(dt),
          horizon(horizon),
          voltage_limit(voltage_limit),
          target_lvdt(target_lvdt),
          position_tolerance(position_tolerance),
          static_friction_penalty(static_friction_penalty),
          internal_substeps(internal_substeps),
          robust_electrical(robust_electrical),
          inductance_rel_uncertainty(inductance_rel_uncertainty),
          pd_blend(pd_blend),
          pd_kp(pd_kp),
          pd_kd(pd_kd),
          pi_ki(pi_ki),
          pi_limit(std::abs(pi_limit)),
          pi_gate_saturation(pi_gate_saturation),
          pi_gate_blocked(pi_gate_blocked),
          pi_gate_error_band(pi_gate_error_band),
          pi_leak_near_setpoint(pi_leak_near_setpoint),
          use_model_integrator(use_model_integrator),
          auto_fc_gain(auto_fc_gain),
          auto_fc_floor(auto_fc_floor),
          auto_fc_cap(auto_fc_cap),
          friction_blend_error_low(friction_blend_error_low),
          friction_blend_error_high(friction_blend_error_high),
          model_bias_limit(std::abs(pi_limit)),
          friction_compensation(0.0),
          output_duty_cycle(false),
          _opt_iters(opt_iters),
          _opt_step(opt_step),
          _opt_eps(opt_eps.has_value() ? *opt_eps : 0.05 * voltage_limit),
          _electrical_alpha(electrical_alpha.has_value() ? *electrical_alpha : 1.0),
          weights_ref(py::none()),
          _state{0.0, 0.0, 0.0},
          _nominal_state{0.0, 0.0, 0.0} {
        Validate_initial_arguments();
        if (weights.is_none()) {
            weights_ref = DefaultWeightsObject();
        } else {
            weights_ref = weights;
        }

        Reset_internal_sequences();
        _apply_motor_parameters(motor, friction_compensation);
    }

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
    bool output_duty_cycle;

    int _opt_iters;
    double _opt_step;
    double _opt_eps;
    double _electrical_alpha;
    double _int_err = 0.0;
    double _u_bias = 0.0;
    std::optional<double> _user_friction_compensation_request;
    double _auto_friction_compensation = 0.0;
    std::optional<double> _active_user_friction_compensation;

    std::optional<double> _last_voltage;
    std::optional<double> _last_measured_position;
    std::optional<double> _last_measurement_time;

    py::object weights_ref;

    std::array<double, 3> _state;
    std::array<double, 3> _nominal_state;
    std::vector<double> _u_seq;

    py::object get_weights() const { return weights_ref; }

    void set_weights(const py::object& weights) {
        if (weights.is_none()) {
            weights_ref = DefaultWeightsObject();
        } else {
            // Ensure it exposes the expected attributes eagerly.
            ParseWeights(weights);
            weights_ref = weights;
        }
    }

    py::object motor() const { return _motor; }
    py::object prediction_models() const { return _prediction_models; }

    py::tuple state_tuple() const {
        return py::make_tuple(_state[0], _state[1], _state[2]);
    }

    py::tuple nominal_state_tuple() const {
        return py::make_tuple(_nominal_state[0], _nominal_state[1], _nominal_state[2]);
    }

    py::tuple u_sequence_tuple() const {
        py::list seq;
        for (double value : _u_seq) {
            seq.append(value);
        }
        return py::tuple(seq);
    }

    void adapt_to_motor(
        const py::object& motor,
        std::optional<double> voltage_limit_arg,
        std::optional<double> friction_compensation_arg) {
        if (voltage_limit_arg.has_value()) {
            if (*voltage_limit_arg <= 0.0) {
                throw py::value_error("voltage_limit must be positive");
            }
            voltage_limit = *voltage_limit_arg;
        }

        _apply_motor_parameters(motor, friction_compensation_arg);

        _nominal_state = _state;
        _u_bias = ClampSymmetric(_u_bias, model_bias_limit);

        if (_last_voltage.has_value()) {
            _last_voltage = ClampSymmetric(*_last_voltage, voltage_limit);
        }

        if (static_cast<int>(_u_seq.size()) != horizon) {
            double base = _last_voltage.value_or(0.0);
            _u_seq.assign(horizon, base);
        }
    }

    void reset(double initial_measurement, double initial_current, double initial_speed) {
        double position = MeasurementToPosition(initial_measurement, motor_params_);
        _state = {initial_current, initial_speed, position};
        _nominal_state = _state;
        _u_seq.assign(horizon, 0.0);
        _last_voltage.reset();
        _last_measured_position = position;
        _last_measurement_time.reset();
        _int_err = 0.0;
        _u_bias = 0.0;
    }

    double update(double time, double measurement) {
        if (!_motor || _motor.is_none()) {
            return 0.0;
        }

        double measured_position = MeasurementToPosition(measurement, motor_params_);
        double normalized_measurement = PositionToMeasurement(measured_position, motor_params_);

        double current = _state[0];
        double estimated_speed = 0.0;

        if (_last_measured_position.has_value() && _last_measurement_time.has_value()) {
            double dt_obs = time - *_last_measurement_time;
            if (dt_obs <= 0.0) {
                throw py::value_error("Controller update time must be strictly increasing");
            }
            double tolerance = std::max(1e-9, 0.25 * dt);
            if (std::abs(dt_obs - dt) > tolerance) {
                throw py::value_error(
                    "Controller update period deviates from configured dt: "
                    "observed=" + std::to_string(dt_obs) + "s expected=" + std::to_string(dt) + "s");
            }
            estimated_speed = (measured_position - *_last_measured_position) / dt_obs;
        }

        _state = {current, estimated_speed, measured_position};
        _last_measured_position = measured_position;
        _last_measurement_time = time;

        if (_last_voltage.has_value()) {
            _nominal_state = PredictNext(_nominal_state, *_last_voltage, motor_params_);
        } else {
            _nominal_state = _state;
        }

        if (use_model_integrator && pi_ki > 0.0) {
            double residual = _state[2] - _nominal_state[2];
            double u_last = _last_voltage.value_or(0.0);
            bool not_saturated = std::abs(u_last) < voltage_limit - 1e-6;
            if (!pi_gate_saturation) {
                not_saturated = true;
            }
            bool allow = not_saturated;
            if (pi_gate_error_band) {
                allow = allow && std::abs(residual) > position_tolerance;
            }
            if (allow) {
                _u_bias += pi_ki * residual * dt;
            } else if (pi_leak_near_setpoint && std::abs(residual) < position_tolerance) {
                _u_bias *= 0.9;
            }
            _u_bias = ClampSymmetric(_u_bias, model_bias_limit);
        }

        double voltage_bias = use_model_integrator ? _u_bias : 0.0;
        const auto sequence = OptimizeSequence(_state, voltage_bias);

        double mpc_voltage = sequence.empty() ? 0.0 : sequence.front();
        if (use_model_integrator) {
            mpc_voltage = ClampSymmetric(mpc_voltage + _u_bias, voltage_limit);
        }

        double position_error = target_lvdt - normalized_measurement;
        double friction_floor = DynamicFrictionCompensation(position_error);

        if (!use_model_integrator && pi_ki > 0.0) {
            double u_last = _last_voltage.value_or(0.0);
            bool not_saturated = std::abs(u_last) < voltage_limit - 1e-6;
            if (!pi_gate_saturation) {
                not_saturated = true;
            }

            bool blocked = false;
            if (pi_gate_blocked) {
                double v_small = motor_params_.stop_speed_threshold * 0.5;
                double u_block = 0.3 * voltage_limit;
                blocked = std::abs(estimated_speed) < v_small && std::abs(u_last) > u_block;
            }

            bool allow_integration = not_saturated && !blocked;
            if (pi_gate_error_band) {
                allow_integration = allow_integration && std::abs(position_error) > friction_blend_error_low;
            }

            if (allow_integration) {
                _int_err += position_error * dt;
            } else if (pi_leak_near_setpoint && std::abs(position_error) < position_tolerance) {
                _int_err *= 0.9;
            }

            double max_int_state = pi_limit / std::max(pi_ki, 1e-9);
            _int_err = ClampSymmetric(_int_err, max_int_state);
        }

        double u_I = use_model_integrator ? _u_bias : ClampSymmetric(pi_ki * _int_err, pi_limit);

        double u_pd = pd_kp * position_error - pd_kd * estimated_speed + u_I;
        double u_nom = pd_blend * mpc_voltage + (1.0 - pd_blend) * u_pd;

        double u_raw = u_nom;

        if (std::abs(_state[1]) < motor_params_.stop_speed_threshold &&
            std::abs(position_error) > position_tolerance &&
            friction_floor > 0.0 &&
            std::abs(u_raw) < friction_floor) {
            u_raw = (position_error >= 0.0) ? friction_floor : -friction_floor;
        }

        double u = ClampSymmetric(u_raw, voltage_limit);
        _state = PredictNext(_state, u, motor_params_);
        _last_voltage = u;

        if (output_duty_cycle) {
            return u / kDutyCycleSupplyVoltage;
        }
        return u;
    }

   private:
    void Validate_initial_arguments() {
        if (dt <= 0.0) {
            throw py::value_error("dt must be positive");
        }
        if (horizon <= 0) {
            throw py::value_error("horizon must be positive");
        }
        if (voltage_limit <= 0.0) {
            throw py::value_error("voltage_limit must be positive");
        }
        if (!(target_lvdt >= -1.0 && target_lvdt <= 1.0)) {
            throw py::value_error("target_lvdt must be within [-1, 1]");
        }
        if (position_tolerance < 0.0) {
            throw py::value_error("position_tolerance must be non-negative");
        }
        if (static_friction_penalty < 0.0) {
            throw py::value_error("static_friction_penalty must be non-negative");
        }
        if (internal_substeps <= 0) {
            throw py::value_error("internal_substeps must be positive");
        }
        if (!robust_electrical) {
            if (!(_electrical_alpha >= 0.0 && _electrical_alpha <= 1.0)) {
                throw py::value_error("electrical_alpha must be within [0, 1]");
            }
        } else {
            if (!(_electrical_alpha >= 0.0 && _electrical_alpha <= 1.0)) {
                throw py::value_error("electrical_alpha must be within [0, 1]");
            }
        }
        if (inductance_rel_uncertainty < 0.0) {
            throw py::value_error("inductance_rel_uncertainty must be non-negative");
        }
        if (!(pd_blend >= 0.0 && pd_blend <= 1.0)) {
            throw py::value_error("pd_blend must be within [0, 1]");
        }
        if (pd_kp <= 0.0) {
            throw py::value_error("pd_kp must be positive");
        }
        if (pd_kd < 0.0) {
            throw py::value_error("pd_kd must be non-negative");
        }
        if (pi_ki < 0.0) {
            throw py::value_error("pi_ki must be non-negative");
        }
        if (pi_limit <= 0.0) {
            throw py::value_error("pi_limit must be positive");
        }
        if (auto_fc_gain <= 0.0) {
            throw py::value_error("auto_fc_gain must be positive");
        }
        if (auto_fc_floor < 0.0) {
            throw py::value_error("auto_fc_floor must be non-negative");
        }
        if (auto_fc_cap.has_value() && *auto_fc_cap <= 0.0) {
            throw py::value_error("auto_fc_cap must be positive when provided");
        }
        if (friction_blend_error_low < 0.0) {
            throw py::value_error("friction_blend_error_low must be non-negative");
        }
        if (friction_blend_error_high <= friction_blend_error_low) {
            throw py::value_error(
                "friction_blend_error_high must be greater than friction_blend_error_low");
        }
        if (_opt_iters <= 0) {
            throw py::value_error("opt_iters must be positive");
        }
        if (_opt_step <= 0.0) {
            throw py::value_error("opt_step must be positive");
        }
        if (_opt_eps <= 0.0) {
            throw py::value_error("opt_eps must be positive");
        }
    }

    void Reset_internal_sequences() {
        _u_seq.assign(horizon, 0.0);
        _state = {0.0, 0.0, 0.0};
        _nominal_state = _state;
    }

    void _apply_motor_parameters(const py::object& motor, std::optional<double> friction_compensation_arg) {
        _motor = motor;
        motor_params_ = MotorParams::FromPython(motor);

        auto friction = _determine_friction_compensation(motor_params_, friction_compensation_arg);
        _auto_friction_compensation = std::get<0>(friction);
        _active_user_friction_compensation = std::get<1>(friction);
        friction_compensation = std::get<2>(friction);

        _prediction_models = BuildPredictionModels(motor);
    }

    std::tuple<double, std::optional<double>, double> _determine_friction_compensation(
        const MotorParams& params,
        std::optional<double> friction_compensation_arg) {
        std::optional<double> friction_value = friction_compensation_arg;

        if (!friction_value.has_value() && _user_friction_compensation_request.has_value()) {
            friction_value = _user_friction_compensation_request;
        } else if (friction_value.has_value()) {
            _user_friction_compensation_request = friction_value;
        }

        std::optional<double> user_value;
        if (friction_value.has_value()) {
            if (*friction_value <= 0.0) {
                throw py::value_error("friction_compensation must be positive");
            }
            user_value = std::min(*friction_value, voltage_limit);
        }

        double base = 0.0;
        if (params.kt != 0.0) {
            base = params.static_friction * params.resistance / params.kt;
        }

        double auto_value = base * auto_fc_gain;
        if (auto_fc_cap.has_value()) {
            auto_value = std::min(auto_value, *auto_fc_cap);
        }
        auto_value = std::max(auto_value, auto_fc_floor);
        auto_value = std::min(auto_value, voltage_limit);

        double effective = user_value.has_value() ? *user_value : auto_value;
        return std::make_tuple(auto_value, user_value, effective);
    }

    py::object BuildPredictionModels(const py::object& motor) {
        py::list models;
        prediction_model_params_.clear();

        models.append(motor);
        prediction_model_params_.push_back(motor_params_);

        if (robust_electrical) {
            return py::tuple(models);
        }

        if (inductance_rel_uncertainty > 0.0) {
            double lower_factor = std::max(1.0 - inductance_rel_uncertainty, 1e-6);
            double upper_factor = 1.0 + inductance_rel_uncertainty;

            if (!IsClose(lower_factor, 1.0, 0.0, 1e-12)) {
                double new_inductance = motor_params_.inductance * lower_factor;
                py::object clone = CloneMotorWithInductance(motor, new_inductance);
                models.append(clone);
                prediction_model_params_.push_back(MotorParams::FromPython(clone));
            }
            if (!IsClose(upper_factor, 1.0, 0.0, 1e-12)) {
                double new_inductance = motor_params_.inductance * upper_factor;
                py::object clone = CloneMotorWithInductance(motor, new_inductance);
                models.append(clone);
                prediction_model_params_.push_back(MotorParams::FromPython(clone));
            }
        }

        return py::tuple(models);
    }

    std::vector<double> OptimizeSequence(const std::array<double, 3>& initial_state, double voltage_bias) {
        std::vector<double> u(horizon, 0.0);

        if (static_cast<int>(_u_seq.size()) == horizon) {
            for (int i = 0; i < horizon; ++i) {
                if (i + 1 < horizon) {
                    u[i] = _u_seq[i + 1];
                } else {
                    u[i] = _u_seq.back();
                }
            }
        } else {
            double base = _last_voltage.value_or(0.0);
            std::fill(u.begin(), u.end(), base);
        }

        for (double& value : u) {
            value = ClampSymmetric(value, voltage_limit);
        }

        for (int iter = 0; iter < _opt_iters; ++iter) {
            for (int i = 0; i < horizon; ++i) {
                double base = u[i];

                double up = ClampSymmetric(base + _opt_eps, voltage_limit);
                u[i] = up;
                auto plus = EvaluateSequence(initial_state, u, voltage_bias);

                double down = ClampSymmetric(base - _opt_eps, voltage_limit);
                u[i] = down;
                auto minus = EvaluateSequence(initial_state, u, voltage_bias);

                u[i] = base;

                double j_plus = std::get<0>(plus);
                double j_minus = std::get<0>(minus);

                if (std::isfinite(j_plus) && std::isfinite(j_minus)) {
                    double grad = (j_plus - j_minus) / (2.0 * _opt_eps);
                    double candidate = base - _opt_step * grad;
                    if (std::isfinite(candidate)) {
                        u[i] = ClampSymmetric(candidate, voltage_limit);
                    }
                }
            }
        }

        _u_seq = u;
        return _u_seq;
    }

    std::tuple<double, std::array<double, 3>, double> EvaluateSequenceSingle(
        const std::array<double, 3>& initial_state,
        const std::vector<double>& sequence,
        const MotorParams& motor,
        double voltage_bias,
        const MPCWeightsValues& weights) {
        std::array<double, 3> state = initial_state;
        double cost = 0.0;
        std::optional<double> previous_voltage = _last_voltage;
        bool have_first = false;
        double first_voltage = 0.0;

        for (std::size_t idx = 0; idx < sequence.size(); ++idx) {
            double voltage = sequence[idx];
            double voltage_eff = voltage + voltage_bias;
            if (voltage_bias != 0.0) {
                voltage_eff = ClampSymmetric(voltage_eff, voltage_limit);
            }

            auto predicted = PredictNext(state, voltage_eff, motor);
            double lvdt = PositionToMeasurement(predicted[2], motor);
            double position_error = target_lvdt - lvdt;

            cost += weights.position * position_error * position_error;
            cost += weights.speed * predicted[1] * predicted[1];

            double delta_voltage = 0.0;
            if (previous_voltage.has_value()) {
                delta_voltage = voltage_eff - *previous_voltage;
            }
            cost += weights.voltage * voltage_eff * voltage_eff;
            cost += weights.delta_voltage * delta_voltage * delta_voltage;
            previous_voltage = voltage_eff;

            if (!have_first) {
                have_first = true;
                first_voltage = voltage_eff;
            }

            double friction_floor = DynamicFrictionCompensation(position_error);
            if (std::abs(predicted[1]) < motor.stop_speed_threshold &&
                std::abs(position_error) > position_tolerance &&
                friction_floor > 0.0 &&
                std::abs(voltage_eff) < friction_floor) {
                cost += static_friction_penalty;
            }

            state = predicted;
        }

        double terminal_measurement = PositionToMeasurement(state[2], motor);
        double terminal_error = target_lvdt - terminal_measurement;
        cost += weights.terminal_position * terminal_error * terminal_error;

        return std::make_tuple(cost, state, first_voltage);
    }

    std::tuple<double, std::array<double, 3>, double> EvaluateSequence(
        const std::array<double, 3>& initial_state,
        const std::vector<double>& sequence,
        double voltage_bias) {
        double worst_cost = -std::numeric_limits<double>::infinity();
        std::array<double, 3> nominal_state = initial_state;
        double nominal_first_voltage = 0.0;

        MPCWeightsValues weights = ParseWeights(weights_ref);

        for (std::size_t index = 0; index < prediction_model_params_.size(); ++index) {
            const auto& motor = prediction_model_params_[index];
            auto result = EvaluateSequenceSingle(initial_state, sequence, motor, voltage_bias, weights);
            double cost = std::get<0>(result);
            if (index == 0) {
                nominal_state = std::get<1>(result);
                nominal_first_voltage = std::get<2>(result);
            }
            if (cost > worst_cost) {
                worst_cost = cost;
            }
        }

        return std::make_tuple(worst_cost, nominal_state, nominal_first_voltage);
    }

    std::array<double, 3> PredictNext(
        const std::array<double, 3>& state,
        double voltage,
        const MotorParams& motor) const {
        std::array<double, 3> result = state;
        double current = result[0];
        double speed = result[1];
        double position = result[2];

        double sub_dt = dt / static_cast<double>(internal_substeps);

        for (int i = 0; i < internal_substeps; ++i) {
            if (robust_electrical) {
                double back_emf = motor.ke * speed;
                double steady_state_current = (voltage - back_emf) / motor.resistance;
                double tau = motor.inductance / motor.resistance;
                double alpha = 1.0;
                if (tau > 0.0 && std::isfinite(tau)) {
                    alpha = 1.0 - std::exp(-sub_dt / tau);
                }
                alpha = std::max(0.0, std::min(alpha, _electrical_alpha));
                current += alpha * (steady_state_current - current);
            } else {
                double di_dt =
                    (voltage - motor.resistance * current - motor.ke * speed) / motor.inductance;
                current += di_dt * sub_dt;
            }

            double electromagnetic_torque = motor.kt * current;
            double spring_torque = motor.SpringTorque(position);
            double available_torque = electromagnetic_torque - spring_torque;

            if (std::abs(speed) < motor.stop_speed_threshold &&
                std::abs(available_torque) <= motor.static_friction) {
                speed = 0.0;
            } else {
                double friction_direction = MotorParams::Sign(speed);
                if (friction_direction == 0.0) {
                    friction_direction = MotorParams::Sign(available_torque);
                }
                double dynamic_friction = motor.coulomb_friction * friction_direction +
                                          motor.viscous_friction * speed;
                double net_torque = available_torque - dynamic_friction;
                double angular_acceleration = net_torque / motor.inertia;
                speed += angular_acceleration * sub_dt;
            }

            position += speed * sub_dt;
        }

        return {current, speed, position};
    }

    double DynamicFrictionCompensation(double position_error) const {
        double e = std::abs(position_error);
        double auto_value = _auto_friction_compensation;
        std::optional<double> user_value = _active_user_friction_compensation;

        if (e <= position_tolerance) {
            return 0.0;
        }

        if (!user_value.has_value()) {
            if (e <= friction_blend_error_low) {
                return 0.0;
            }
            if (e >= friction_blend_error_high) {
                return std::min(auto_value, voltage_limit);
            }
            double alpha = (e - friction_blend_error_low) /
                           (friction_blend_error_high - friction_blend_error_low);
            return std::min(alpha * auto_value, voltage_limit);
        }

        if (e <= friction_blend_error_low) {
            return std::min(auto_value, voltage_limit);
        }
        if (e >= friction_blend_error_high) {
            return std::min(*user_value, voltage_limit);
        }

        double alpha = (e - friction_blend_error_low) /
                       (friction_blend_error_high - friction_blend_error_low);
        double blended = (1.0 - alpha) * auto_value + alpha * (*user_value);
        return std::min(blended, voltage_limit);
    }

    py::object _motor;
    py::object _prediction_models;
    MotorParams motor_params_;
    std::vector<MotorParams> prediction_model_params_;
};

}  // namespace motor_model

