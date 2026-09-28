#pragma once
// =====================================================================
// Profiler.hpp - 轻量级 RAII 性能分析工具（header-only, 无需 link）
//
// 用法：
//   #include "Profiler.hpp"
//   {
//       Profiler::ScopedTimer t("StageName");  // 在作用域结束时自动打点
//       ...do something...
//   }
//
// 后台定期打印汇总：
//   Profiler::Instance().StartAutoReport(std::chrono::seconds(5));
//
// 手动打印：
//   Profiler::Instance().PrintSummary();
//   Profiler::Instance().DumpJson();         // 给前端 /api/profile 用
// =====================================================================

#include <chrono>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <thread>
#include <atomic>
#include <sstream>

class TimingStats {
public:
    void Record(double ms) {
        std::lock_guard<std::mutex> lock(m_);
        ++count_;
        sum_ += ms;
        if (ms < min_) min_ = ms;
        if (ms > max_) max_ = ms;
        last_ = ms;
    }
    struct Snap {
        uint64_t count;
        double avg, mn, mx, last;
    };
    Snap Snapshot() const {
        std::lock_guard<std::mutex> lock(m_);
        return {count_, count_ ? sum_ / count_ : 0.0, min_, max_, last_};
    }
    void Reset() {
        std::lock_guard<std::mutex> lock(m_);
        count_ = 0; sum_ = 0;
        min_ = 1e9; max_ = 0; last_ = 0;
    }
private:
    mutable std::mutex m_;
    uint64_t count_ = 0;
    double sum_ = 0;
    double min_ = 1e9;
    double max_ = 0;
    double last_ = 0;
};

class Profiler {
public:
    static Profiler& Instance() {
        static Profiler p;
        return p;
    }

    // 记录一次耗时（毫秒）
    void Record(const std::string& name, double ms) {
        std::lock_guard<std::mutex> lock(m_);
        stats_[name].Record(ms);
    }

    void Reset() {
        std::lock_guard<std::mutex> lock(m_);
        stats_.clear();
    }

    // 打印文本汇总
    void PrintSummary() const {
        std::lock_guard<std::mutex> lock(m_);
        if (stats_.empty()) {
            std::printf("[Profiler] (no samples yet)\n");
            return;
        }
        std::printf("\n================== [Profiler] Summary ==================\n");
        std::printf("%-28s %8s %10s %10s %10s %10s\n",
                    "Stage", "Count", "Avg(ms)", "Min", "Max", "Last");
        std::printf("--------------------------------------------------------\n");
        for (auto& [name, s] : stats_) {
            auto snap = s.Snapshot();
            std::printf("%-28s %8llu %10.2f %10.2f %10.2f %10.2f\n",
                        name.c_str(),
                        (unsigned long long)snap.count,
                        snap.avg, snap.mn, snap.mx, snap.last);
        }
        std::printf("========================================================\n\n");
    }

    // 导出 JSON（给 Web 前端或 /api/profile 接口用）
    std::string DumpJson() const {
        std::lock_guard<std::mutex> lock(m_);
        std::ostringstream os;
        os << "{\"stages\":[";
        bool first = true;
        for (auto& [name, s] : stats_) {
            auto snap = s.Snapshot();
            if (!first) os << ",";
            first = false;
            os << "{\"name\":\"" << name << "\","
               << "\"count\":" << snap.count << ","
               << "\"avg\":" << snap.avg << ","
               << "\"min\":" << snap.mn << ","
               << "\"max\":" << snap.mx << ","
               << "\"last\":" << snap.last << "}";
        }
        os << "]}";
        return os.str();
    }

    // 启动后台定期打印线程（启动一次即可）
    void StartAutoReport(std::chrono::seconds interval) {
        if (reporter_running_.exchange(true)) return;  // 已启动则跳过
        reporter_thread_ = std::thread([this, interval]() {
            while (reporter_running_) {
                std::this_thread::sleep_for(interval);
                if (!reporter_running_) break;
                PrintSummary();
            }
        });
        reporter_thread_.detach();
        std::printf("[Profiler] Auto report started, interval=%lld s\n",
                  (long long)interval.count());
    }

    void StopAutoReport() {
        reporter_running_ = false;
    }

private:
    Profiler() = default;
    mutable std::mutex m_;
    std::unordered_map<std::string, TimingStats> stats_;
    std::atomic<bool> reporter_running_{false};
    std::thread reporter_thread_;
};

// RAII 计时器：构造时开始，析构时自动上报
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name)
        : name_(name), start_(std::chrono::steady_clock::now()) {}

    // 可手动停止并提前上报（用于不想等到析构的场景）
    double Stop() {
        if (stopped_) return last_ms_;
        stopped_ = true;
        auto end = std::chrono::steady_clock::now();
        last_ms_ = std::chrono::duration<double, std::milli>(end - start_).count();
        Profiler::Instance().Record(name_, last_ms_);
        return last_ms_;
    }

    ~ScopedTimer() { if (!stopped_) Stop(); }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
    double last_ms_ = 0.0;
    bool stopped_ = false;
};