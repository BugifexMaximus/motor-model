#include "continuous_mpc_manager.h"

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace motor_model {

struct ContMPCControllerManager::StepTask {
    double time{0.0};
    double measurement{0.0};
    StepCallback callback;
    std::promise<double> promise;
};

ContMPCControllerManager::ContMPCControllerManager(ContMPCController controller)
    : ContMPCControllerManager(
          std::make_shared<ContMPCController>(std::move(controller))) {}

ContMPCControllerManager::ContMPCControllerManager(
    std::shared_ptr<ContMPCController> controller)
    : controller_(std::move(controller)) {
    if (!controller_) {
        throw std::invalid_argument("ContMPCControllerManager requires a controller");
    }
}

ContMPCControllerManager::~ContMPCControllerManager() { Stop(); }

void ContMPCControllerManager::StartRealTime(
    MeasurementProvider provider,
    StepCallback callback,
    double frequency_hz) {
    if (!provider) {
        throw std::invalid_argument("Measurement provider must be callable");
    }

    std::unique_lock<std::mutex> lock(state_mutex_);
    StopLocked(lock);

    realtime_provider_ = std::move(provider);
    realtime_callback_ = std::move(callback);
    realtime_frequency_hz_ = EffectiveFrequency(frequency_hz);
    stop_requested_.store(false, std::memory_order_release);
    mode_ = Mode::kRealTime;
    worker_ = std::thread(&ContMPCControllerManager::RealTimeLoop, this);
}

std::future<double> ContMPCControllerManager::SubmitStep(
    double time,
    double measurement,
    StepCallback callback) {
    std::unique_lock<std::mutex> lock(state_mutex_);
    if (mode_ == Mode::kRealTime) {
        throw std::logic_error(
            "Cannot submit discrete steps while running in real-time mode");
    }
    if (mode_ == Mode::kIdle) {
        stop_requested_.store(false, std::memory_order_release);
        mode_ = Mode::kTriggered;
        worker_ = std::thread(&ContMPCControllerManager::TriggeredLoop, this);
    }
    lock.unlock();

    auto task = std::make_unique<StepTask>();
    auto future = task->promise.get_future();
    task->time = time;
    task->measurement = measurement;
    task->callback = std::move(callback);

    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        queue_.push_back(std::move(task));
    }
    cv_.notify_one();
    return future;
}

void ContMPCControllerManager::Stop() {
    std::unique_lock<std::mutex> lock(state_mutex_);
    StopLocked(lock);
}

void ContMPCControllerManager::StopLocked(std::unique_lock<std::mutex>& lock) {
    if (mode_ == Mode::kIdle) {
        return;
    }

    stop_requested_.store(true, std::memory_order_release);
    const bool was_triggered = mode_ == Mode::kTriggered;
    std::vector<std::unique_ptr<StepTask>> pending_tasks;
    if (was_triggered) {
        {
            std::lock_guard<std::mutex> queue_lock(queue_mutex_);
            while (!queue_.empty()) {
                pending_tasks.emplace_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }
        cv_.notify_all();
    }

    std::thread worker = std::move(worker_);
    lock.unlock();
    if (worker.joinable()) {
        worker.join();
    }

    if (was_triggered) {
        for (auto& task : pending_tasks) {
            if (!task) {
                continue;
            }
            try {
                throw std::runtime_error(
                    "ContMPCControllerManager stopped before executing step");
            } catch (...) {
                task->promise.set_exception(std::current_exception());
            }
        }
    }

    lock.lock();
    stop_requested_.store(false, std::memory_order_release);
    mode_ = Mode::kIdle;
    realtime_provider_ = MeasurementProvider{};
    realtime_callback_ = StepCallback{};
    realtime_frequency_hz_ = 0.0;
}

double ContMPCControllerManager::EffectiveFrequency(double requested_frequency_hz) const {
    if (requested_frequency_hz > 0.0) {
        return requested_frequency_hz;
    }
    if (controller_->dt <= 0.0) {
        throw std::invalid_argument(
            "Controller dt must be positive to compute real-time frequency");
    }
    return 1.0 / controller_->dt;
}

void ContMPCControllerManager::RealTimeLoop() {
    using Clock = std::chrono::steady_clock;
    const double frequency_hz = realtime_frequency_hz_;
    const auto period = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(1.0 / frequency_hz));
    const double period_seconds = std::chrono::duration<double>(period).count();
    auto next_deadline = Clock::now();

    while (!stop_requested_.load(std::memory_order_acquire)) {
        next_deadline += period;
        std::pair<double, double> input{};
        try {
            input = realtime_provider_();
        } catch (const std::exception& e) {
            std::cerr << "[ContMPCControllerManager] measurement provider threw: "
                      << e.what() << std::endl;
            break;
        } catch (...) {
            std::cerr << "[ContMPCControllerManager] measurement provider threw an unknown exception"
                      << std::endl;
            break;
        }

        double control = 0.0;
        try {
            control = controller_->update(input.first, input.second);
        } catch (const std::exception& e) {
            std::cerr << "[ContMPCControllerManager] controller update failed: "
                      << e.what() << std::endl;
            break;
        } catch (...) {
            std::cerr << "[ContMPCControllerManager] controller update failed with unknown exception"
                      << std::endl;
            break;
        }

        if (realtime_callback_) {
            try {
                realtime_callback_(input.first, input.second, control);
            } catch (const std::exception& e) {
                std::cerr << "[ContMPCControllerManager] realtime callback threw: "
                          << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[ContMPCControllerManager] realtime callback threw an unknown exception"
                          << std::endl;
            }
        }

        const auto now = Clock::now();
        if (now > next_deadline) {
            const auto overrun = std::chrono::duration_cast<std::chrono::duration<double>>(now - next_deadline);
            std::cerr << "[ContMPCControllerManager] step overran by "
                      << overrun.count() * 1e3
                      << " ms while targeting period "
                      << period_seconds * 1e3 << " ms" << std::endl;
        } else {
            std::this_thread::sleep_until(next_deadline);
        }
    }
}

void ContMPCControllerManager::TriggeredLoop() {
    while (true) {
        std::unique_ptr<StepTask> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this]() {
                return stop_requested_.load(std::memory_order_acquire) || !queue_.empty();
            });
            if (stop_requested_.load(std::memory_order_acquire) && queue_.empty()) {
                break;
            }
            if (!queue_.empty()) {
                task = std::move(queue_.front());
                queue_.pop_front();
            }
        }

        if (!task) {
            continue;
        }

        try {
            double control = controller_->update(task->time, task->measurement);
            if (task->callback) {
                try {
                    task->callback(task->time, task->measurement, control);
                } catch (const std::exception& e) {
                    std::cerr << "[ContMPCControllerManager] triggered callback threw: "
                              << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "[ContMPCControllerManager] triggered callback threw an unknown exception"
                              << std::endl;
                }
            }
            task->promise.set_value(control);
        } catch (...) {
            task->promise.set_exception(std::current_exception());
        }
    }
}

}  // namespace motor_model

