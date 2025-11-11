#include "continuous_mpc_core.h"

namespace motor_model {

namespace {

constexpr double kMinPositive = 1e-12;

}  // namespace

ContMPCController::ContMPCController(
    MotorParams motor,
    double dt,
    int horizon,
    double voltage_limit,
    double target_lvdt,
    MPCWeightsValues weights,
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
    std::optional<double> opt_eps)
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
      _opt_iters(opt_iters),
      _opt_step(opt_step),
      _opt_eps(opt_eps.has_value() ? *opt_eps : 0.05 * voltage_limit),
      _electrical_alpha(electrical_alpha.has_value() ? *electrical_alpha : 1.0),
      motor_params_(motor),
      weights_(weights) {
    Validate_initial_arguments();
    Reset_internal_sequences();
    Apply_motor_parameters(motor, friction_compensation);
}

void ContMPCController::Validate_initial_arguments() {
    if (dt <= 0.0) {
        throw std::invalid_argument("dt must be positive");
    }
    if (horizon <= 0) {
        throw std::invalid_argument("horizon must be positive");
    }
    if (voltage_limit <= 0.0) {
        throw std::invalid_argument("voltage_limit must be positive");
    }
    if (!(target_lvdt >= -1.0 && target_lvdt <= 1.0)) {
        throw std::invalid_argument("target_lvdt must be within [-1, 1]");
    }
    if (position_tolerance < 0.0) {
        throw std::invalid_argument("position_tolerance must be non-negative");
    }
    if (static_friction_penalty < 0.0) {
        throw std::invalid_argument("static_friction_penalty must be non-negative");
    }
    if (internal_substeps <= 0) {
        throw std::invalid_argument("internal_substeps must be positive");
    }
    if (_electrical_alpha < 0.0 || _electrical_alpha > 1.0) {
        throw std::invalid_argument("electrical_alpha must be within [0, 1]");
    }
    if (inductance_rel_uncertainty < 0.0) {
        throw std::invalid_argument("inductance_rel_uncertainty must be non-negative");
    }
    if (!(pd_blend >= 0.0 && pd_blend <= 1.0)) {
        throw std::invalid_argument("pd_blend must be within [0, 1]");
    }
    if (pd_kp <= 0.0) {
        throw std::invalid_argument("pd_kp must be positive");
    }
    if (pd_kd < 0.0) {
        throw std::invalid_argument("pd_kd must be non-negative");
    }
    if (pi_ki < 0.0) {
        throw std::invalid_argument("pi_ki must be non-negative");
    }
    if (pi_limit <= 0.0) {
        throw std::invalid_argument("pi_limit must be positive");
    }
    if (auto_fc_gain <= 0.0) {
        throw std::invalid_argument("auto_fc_gain must be positive");
    }
    if (auto_fc_floor < 0.0) {
        throw std::invalid_argument("auto_fc_floor must be non-negative");
    }
    if (auto_fc_cap.has_value() && *auto_fc_cap <= 0.0) {
        throw std::invalid_argument("auto_fc_cap must be positive when provided");
    }
    if (friction_blend_error_low < 0.0) {
        throw std::invalid_argument("friction_blend_error_low must be non-negative");
    }
    if (friction_blend_error_high <= friction_blend_error_low) {
        throw std::invalid_argument(
            "friction_blend_error_high must be greater than friction_blend_error_low");
    }
    if (_opt_iters <= 0) {
        throw std::invalid_argument("opt_iters must be positive");
    }
    if (_opt_step <= 0.0) {
        throw std::invalid_argument("opt_step must be positive");
    }
    if (_opt_eps <= 0.0) {
        throw std::invalid_argument("opt_eps must be positive");
    }
}

void ContMPCController::Reset_internal_sequences() {
    _u_seq.assign(static_cast<std::size_t>(horizon), 0.0);
    _state = {0.0, 0.0, 0.0};
    _nominal_state = _state;
}

void ContMPCController::Apply_motor_parameters(
    const MotorParams& motor,
    std::optional<double> friction_compensation_arg) {
    motor_params_ = motor;

    auto friction = Determine_friction_compensation(motor_params_, friction_compensation_arg);
    _auto_friction_compensation = std::get<0>(friction);
    _active_user_friction_compensation = std::get<1>(friction);
    friction_compensation = std::get<2>(friction);

    prediction_model_params_ = Build_prediction_models(motor_params_);
}

std::tuple<double, std::optional<double>, double>
ContMPCController::Determine_friction_compensation(
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
            throw std::invalid_argument("friction_compensation must be positive");
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

std::vector<MotorParams> ContMPCController::Build_prediction_models(const MotorParams& motor) {
    std::vector<MotorParams> models;
    models.push_back(motor);

    if (robust_electrical) {
        return models;
    }

    if (inductance_rel_uncertainty > 0.0) {
        double lower_factor = std::max(1.0 - inductance_rel_uncertainty, 1e-6);
        double upper_factor = 1.0 + inductance_rel_uncertainty;

        if (!IsClose(lower_factor, 1.0, 0.0, kMinPositive)) {
            models.push_back(motor.WithInductance(motor.inductance * lower_factor));
        }
        if (!IsClose(upper_factor, 1.0, 0.0, kMinPositive)) {
            models.push_back(motor.WithInductance(motor.inductance * upper_factor));
        }
    }

    return models;
}

void ContMPCController::adapt_to_motor(
    const MotorParams& motor,
    std::optional<double> voltage_limit_arg,
    std::optional<double> friction_compensation_arg) {
    if (voltage_limit_arg.has_value()) {
        if (*voltage_limit_arg <= 0.0) {
            throw std::invalid_argument("voltage_limit must be positive");
        }
        voltage_limit = *voltage_limit_arg;
    }

    Apply_motor_parameters(motor, friction_compensation_arg);

    _nominal_state = _state;
    _u_bias = ClampSymmetric(_u_bias, model_bias_limit);

    if (_last_voltage.has_value()) {
        _last_voltage = ClampSymmetric(*_last_voltage, voltage_limit);
    }

    if (static_cast<int>(_u_seq.size()) != horizon) {
        double base = _last_voltage.value_or(0.0);
        _u_seq.assign(static_cast<std::size_t>(horizon), base);
    }
}

void ContMPCController::reset(
    double initial_measurement,
    double initial_current,
    double initial_speed) {
    double position = MeasurementToPosition(initial_measurement, motor_params_);
    _state = {initial_current, initial_speed, position};
    _nominal_state = _state;
    _u_seq.assign(static_cast<std::size_t>(horizon), 0.0);
    _last_voltage.reset();
    _last_measured_position = position;
    _last_measurement_time.reset();
    _int_err = 0.0;
    _u_bias = 0.0;
}

double ContMPCController::update(double time, double measurement) {
    double measured_position = MeasurementToPosition(measurement, motor_params_);
    double normalized_measurement = PositionToMeasurement(measured_position, motor_params_);

    double current = _state[0];
    double estimated_speed = 0.0;

    if (_last_measured_position.has_value() && _last_measurement_time.has_value()) {
        double dt_obs = time - *_last_measurement_time;
        if (dt_obs <= 0.0) {
            throw std::invalid_argument("Controller update time must be strictly increasing");
        }
        double tolerance = std::max(1e-9, 0.25 * dt);
        if (std::abs(dt_obs - dt) > tolerance) {
            throw std::invalid_argument(
                "Controller update period deviates from configured dt: observed=" +
                std::to_string(dt_obs) + "s expected=" + std::to_string(dt) + "s");
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

        double max_int_state = pi_limit / std::max(pi_ki, kMinPositive);
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

std::vector<double> ContMPCController::OptimizeSequence(
    const std::array<double, 3>& initial_state,
    double voltage_bias) {
    std::vector<double> u(static_cast<std::size_t>(horizon), 0.0);

    if (static_cast<int>(_u_seq.size()) == horizon) {
        for (int i = 0; i < horizon; ++i) {
            if (i + 1 < horizon) {
                u[static_cast<std::size_t>(i)] = _u_seq[static_cast<std::size_t>(i + 1)];
            } else {
                u[static_cast<std::size_t>(i)] = _u_seq.back();
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
            std::size_t index = static_cast<std::size_t>(i);
            double base = u[index];

            double up = ClampSymmetric(base + _opt_eps, voltage_limit);
            u[index] = up;
            auto plus = EvaluateSequence(initial_state, u, voltage_bias);

            double down = ClampSymmetric(base - _opt_eps, voltage_limit);
            u[index] = down;
            auto minus = EvaluateSequence(initial_state, u, voltage_bias);

            u[index] = base;

            double j_plus = std::get<0>(plus);
            double j_minus = std::get<0>(minus);

            if (std::isfinite(j_plus) && std::isfinite(j_minus)) {
                double grad = (j_plus - j_minus) / (2.0 * _opt_eps);
                double candidate = base - _opt_step * grad;
                if (std::isfinite(candidate)) {
                    u[index] = ClampSymmetric(candidate, voltage_limit);
                }
            }
        }
    }

    _u_seq = u;
    return _u_seq;
}

std::tuple<double, std::array<double, 3>, double>
ContMPCController::EvaluateSequenceSingle(
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

    for (double voltage : sequence) {
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

std::tuple<double, std::array<double, 3>, double>
ContMPCController::EvaluateSequence(
    const std::array<double, 3>& initial_state,
    const std::vector<double>& sequence,
    double voltage_bias) {
    double worst_cost = -std::numeric_limits<double>::infinity();
    std::array<double, 3> nominal_state = initial_state;
    double nominal_first_voltage = 0.0;

    for (std::size_t index = 0; index < prediction_model_params_.size(); ++index) {
        const auto& motor = prediction_model_params_[index];
        auto result = EvaluateSequenceSingle(initial_state, sequence, motor, voltage_bias, weights_);
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

std::array<double, 3> ContMPCController::PredictNext(
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

double ContMPCController::DynamicFrictionCompensation(double position_error) const {
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

}  // namespace motor_model

