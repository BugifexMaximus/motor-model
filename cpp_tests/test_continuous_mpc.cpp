#include "continuous_mpc_core.h"
#include "continuous_mpc_manager.h"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <mutex>
#include <optional>

namespace {

motor_model::MotorParams MakeTestMotor() {
    motor_model::MotorParams motor;
    motor.resistance = 24.0;
    motor.inductance = 14e-3;
    motor.kv = 8.5;
    motor.ke = 0.045;
    motor.kt = 0.045;
    motor.inertia = 4.8e-4;
    motor.viscous_friction = 1.9e-4;
    motor.coulomb_friction = 2.1e-2;
    motor.static_friction = 2.4e-2;
    motor.stop_speed_threshold = 1e-4;
    motor.spring_constant = 9.5e-4;
    motor.spring_compression_ratio = 0.4;
    motor.lvdt_full_scale = 0.1;
    motor.integration_substeps = 4;
    return motor;
}

motor_model::MPCWeightsValues MakeTestWeights() {
    motor_model::MPCWeightsValues weights;
    weights.position = 250.0;
    weights.speed = 0.6;
    weights.voltage = 0.03;
    weights.delta_voltage = 0.2;
    weights.terminal_position = 500.0;
    return weights;
}

motor_model::ContMPCController MakeController(
    const motor_model::MotorParams& motor,
    bool robust_electrical,
    double inductance_rel_uncertainty) {
    auto weights = MakeTestWeights();
    return motor_model::ContMPCController(
        motor,
        /*dt=*/0.01,
        /*horizon=*/4,
        /*voltage_limit=*/6.5,
        /*target_lvdt=*/0.0,
        weights,
        /*position_tolerance=*/0.01,
        /*static_friction_penalty=*/30.0,
        /*friction_compensation=*/std::nullopt,
        /*auto_fc_gain=*/1.2,
        /*auto_fc_floor=*/0.02,
        /*auto_fc_cap=*/std::optional<double>(),
        /*friction_blend_error_low=*/0.05,
        /*friction_blend_error_high=*/0.3,
        /*internal_substeps=*/8,
        /*robust_electrical=*/robust_electrical,
        /*electrical_alpha=*/0.6,
        /*inductance_rel_uncertainty=*/inductance_rel_uncertainty,
        /*pd_blend=*/0.55,
        /*pd_kp=*/4.0,
        /*pd_kd=*/0.3,
        /*pi_ki=*/0.4,
        /*pi_limit=*/2.5,
        /*pi_gate_saturation=*/true,
        /*pi_gate_blocked=*/true,
        /*pi_gate_error_band=*/true,
        /*pi_leak_near_setpoint=*/true,
        /*use_model_integrator=*/false,
        /*opt_iters=*/5,
        /*opt_step=*/0.1,
        /*opt_eps=*/std::optional<double>());
}

}  // namespace

TEST(MotorParamsTest, SpringTorqueRespectsCompressionRatio) {
    motor_model::MotorParams params = MakeTestMotor();
    params.spring_constant = 2.0;
    params.spring_compression_ratio = 0.5;

    EXPECT_DOUBLE_EQ(params.SpringTorque(0.1), 0.2);
    EXPECT_DOUBLE_EQ(params.SpringTorque(-0.1), -0.1);
    params.spring_zero_position = 0.2;
    EXPECT_DOUBLE_EQ(params.SpringTorque(0.3), 0.2);
    EXPECT_DOUBLE_EQ(params.SpringTorque(0.0), -0.2);
    params.spring_constant = 0.0;
    EXPECT_DOUBLE_EQ(params.SpringTorque(0.5), 0.0);
}

TEST(ContMPCControllerTest, PredictionModelsExpandWhenRobustDisabled) {
    motor_model::MotorParams motor = MakeTestMotor();
    auto controller = MakeController(motor, /*robust_electrical=*/false, /*inductance_rel_uncertainty=*/0.25);

    ASSERT_EQ(controller.prediction_models().size(), 3u);
    EXPECT_NEAR(controller.prediction_models()[1].inductance, motor.inductance * 0.75, 1e-12);
    EXPECT_NEAR(controller.prediction_models()[2].inductance, motor.inductance * 1.25, 1e-12);

    auto robust_controller = MakeController(motor, /*robust_electrical=*/true, /*inductance_rel_uncertainty=*/0.25);
    EXPECT_EQ(robust_controller.prediction_models().size(), 1u);
}

TEST(ContMPCControllerTest, ResetInitializesStateAndNominalState) {
    auto controller = MakeController(MakeTestMotor(), /*robust_electrical=*/true, /*inductance_rel_uncertainty=*/0.0);
    controller.reset(/*initial_measurement=*/0.25, /*initial_current=*/0.3, /*initial_speed=*/-0.2);

    const auto& state = controller.state();
    const auto& nominal = controller.nominal_state();
    EXPECT_NEAR(state[0], 0.3, 1e-12);
    EXPECT_NEAR(state[1], -0.2, 1e-12);
    EXPECT_NEAR(state[2], 0.025, 1e-12);
    EXPECT_DOUBLE_EQ(state[0], nominal[0]);
    EXPECT_DOUBLE_EQ(state[1], nominal[1]);
    EXPECT_DOUBLE_EQ(state[2], nominal[2]);
}

TEST(ContMPCControllerTest, UpdateAtSetpointProducesZeroVoltage) {
    auto controller = MakeController(MakeTestMotor(), /*robust_electrical=*/true, /*inductance_rel_uncertainty=*/0.0);
    controller.reset(/*initial_measurement=*/0.0, /*initial_current=*/0.0, /*initial_speed=*/0.0);

    double output = controller.update(/*time=*/0.01, /*measurement=*/0.0);
    EXPECT_NEAR(output, 0.0, 1e-6);

    const auto& state = controller.state();
    EXPECT_NEAR(state[0], 0.0, 1e-6);
    EXPECT_NEAR(state[1], 0.0, 1e-6);
    EXPECT_NEAR(state[2], 0.0, 1e-6);
}

TEST(ContMPCControllerTest, UpdateRequiresMonotonicTime) {
    auto controller = MakeController(MakeTestMotor(), /*robust_electrical=*/true, /*inductance_rel_uncertainty=*/0.0);
    controller.reset(/*initial_measurement=*/0.0, /*initial_current=*/0.0, /*initial_speed=*/0.0);

    EXPECT_NO_THROW(controller.update(/*time=*/0.01, /*measurement=*/0.0));
    EXPECT_THROW(controller.update(/*time=*/0.01, /*measurement=*/0.0), std::invalid_argument);
}

TEST(ContMPCControllerTest, AdaptToMotorUpdatesParameters) {
    motor_model::MotorParams motor = MakeTestMotor();
    auto controller = MakeController(motor, /*robust_electrical=*/true, /*inductance_rel_uncertainty=*/0.0);
    controller.reset(/*initial_measurement=*/0.0, /*initial_current=*/0.0, /*initial_speed=*/0.0);

    motor_model::MotorParams updated = motor;
    updated.resistance = 20.0;
    updated.inductance = 12e-3;
    updated.kt = 0.05;

    controller.adapt_to_motor(updated, /*voltage_limit_arg=*/6.0, /*friction_compensation_arg=*/0.12);

    EXPECT_NEAR(controller.motor_params().resistance, 20.0, 1e-12);
    EXPECT_NEAR(controller.motor_params().inductance, 12e-3, 1e-12);
    EXPECT_NEAR(controller.friction_compensation, 0.12, 1e-12);
    EXPECT_DOUBLE_EQ(controller.voltage_limit, 6.0);
    EXPECT_EQ(controller.u_sequence().size(), 4u);
}

TEST(ContMPCControllerManagerTest, RealTimeLoopInvokesCallback) {
    auto controller = MakeController(MakeTestMotor(), /*robust_electrical=*/true, /*inductance_rel_uncertainty=*/0.0);
    controller.reset(/*initial_measurement=*/0.0, /*initial_current=*/0.0, /*initial_speed=*/0.0);

    motor_model::ContMPCControllerManager manager(std::move(controller));

    std::mutex mutex;
    std::condition_variable cv;
    int callback_count = 0;
    std::atomic<int> provider_calls{0};

    manager.StartRealTime(
        [&provider_calls]() {
            int call = ++provider_calls;
            return std::make_pair(static_cast<double>(call) * 0.01, 0.0);
        },
        [&mutex, &cv, &callback_count](double /*time*/, double /*measurement*/, double /*control*/) {
            std::lock_guard<std::mutex> lock(mutex);
            ++callback_count;
            cv.notify_all();
        },
        /*frequency_hz=*/100.0);

    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, std::chrono::milliseconds(200), [&callback_count]() { return callback_count >= 3; });
    }

    manager.Stop();

    EXPECT_GE(callback_count, 3);
}

TEST(ContMPCControllerManagerTest, TriggeredStepProvidesFutureAndCallback) {
    auto controller = MakeController(MakeTestMotor(), /*robust_electrical=*/true, /*inductance_rel_uncertainty=*/0.0);
    controller.reset(/*initial_measurement=*/0.0, /*initial_current=*/0.0, /*initial_speed=*/0.0);

    motor_model::ContMPCControllerManager manager(std::move(controller));

    std::mutex mutex;
    std::condition_variable cv;
    bool callback_called = false;

    auto future = manager.SubmitStep(
        /*time=*/0.01,
        /*measurement=*/0.0,
        [&mutex, &cv, &callback_called](double /*time*/, double /*measurement*/, double /*control*/) {
            std::lock_guard<std::mutex> lock(mutex);
            callback_called = true;
            cv.notify_all();
        });

    double result = future.get();

    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, std::chrono::milliseconds(100), [&callback_called]() { return callback_called; });
    }

    manager.Stop();

    EXPECT_TRUE(callback_called);
    EXPECT_TRUE(std::isfinite(result));
}

