#pragma once

#include "continuous_mpc_core.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace motor_model {

class ContMPCControllerManager {
   public:
    using StepCallback = std::function<void(double time, double measurement, double control)>;
    using MeasurementProvider = std::function<std::pair<double, double>()>;

    explicit ContMPCControllerManager(ContMPCController controller);
    explicit ContMPCControllerManager(std::shared_ptr<ContMPCController> controller);
    ~ContMPCControllerManager();

    ContMPCControllerManager(const ContMPCControllerManager&) = delete;
    ContMPCControllerManager& operator=(const ContMPCControllerManager&) = delete;

    void StartRealTime(MeasurementProvider provider, StepCallback callback, double frequency_hz = 0.0);

    std::future<double> SubmitStep(double time, double measurement, StepCallback callback = StepCallback{});

    void Stop();

   private:
    struct StepTask;

    enum class Mode { kIdle, kRealTime, kTriggered };

    double EffectiveFrequency(double requested_frequency_hz) const;

    void StopLocked(std::unique_lock<std::mutex>& lock);

    void RealTimeLoop();
    void TriggeredLoop();

    std::shared_ptr<ContMPCController> controller_;
    std::atomic<bool> stop_requested_{false};
    mutable std::mutex state_mutex_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::deque<std::unique_ptr<StepTask>> queue_;
    std::thread worker_;
    Mode mode_{Mode::kIdle};
    MeasurementProvider realtime_provider_;
    StepCallback realtime_callback_;
    double realtime_frequency_hz_{0.0};
};

}  // namespace motor_model

