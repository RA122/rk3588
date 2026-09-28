#pragma once
#include "RKNNInferEngine.hpp"
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

struct AlertEvent {
  std::string timestamp;     // 标准格式时间戳 (如: "2026-09-17 15:40:00")
  std::string raw_timestamp; // 文件名格式时间戳 (如: "20260917_154000")
  std::string cam_id;        // 摄像头通道 ID (如: "D1")
  std::string cam_name;      // 摄像头物理名称 (如: "通道 D1")
  std::string
      event_type; // 主类别英文名（去重后的第一项，向后兼容）(如: "landslide")
  std::vector<std::string> event_types; // 所有检测到的唯一类别英文名（按出现顺序），如
                                        // ["landslide", "stone"]
  std::string event_name_cn; // 主类别中文名（向后兼容）
  std::vector<std::string>
      event_names_cn;          // 所有类别中文名，与 event_types 一一对应
  std::string image_path;      // 本地抓拍文件完整路径
  std::string image_name;      // 抓拍图片文件名 (如:
                               // "D1_landslide+stone_20260917_154000.jpg")
  float max_confidence = 0.0f; // 最高置信度
  std::vector<DetectResult> targets; // 详细目标列表
};

struct TrackedObject {
  int id;               // 追踪分配的唯一 ID
  int class_id;         // 类别 ID
  cv::Point2f centroid; // 最新质心坐标
  cv::Point2f
      start_centroid; // 最初始质心坐标 (用于计算绝对位移，过滤原生框的抖动)
  cv::Rect start_box; // 最初始的边界框 (用于计算 IoU，过滤大目标及手抖)
  std::vector<cv::Rect> history_boxes; // 历史框记录缓冲 (用于判定是否已经静止)
  int active_frames_count = 0;         // 持续被追踪的帧数
  int moving_frames_count = 0;         // (废弃，可随时移除)
  int lost_frames_count = 0;           // 丢失追踪的连续帧数
  bool has_triggered_static_alarm = false; // 是否已触发过静态落地首报
};

// 两级检测的第二级 —— "二次确认"回调.
// 常驻推理跑在降采样帧上(省 CPU), 代价是杂物在低分辨率下容易和目标混淆.
// 所以在真正落盘/推送之前, 用高清帧再检一次:
//   返回 true  → 确认通过, out_full_frame 是高清帧, out_results 是它上面检出的框
//   返回 false → 判定为误检, 本次告警直接丢弃
// 注意返回的帧与框必须是同一坐标系, 这样存图时不需要任何换算.
using ConfirmCallback = std::function<bool(
    const std::string& cam_id,
    const std::vector<DetectResult>& low_res_results,
    cv::Mat& out_full_frame,
    std::vector<DetectResult>& out_results)>;

class AlertService {
public:
  AlertService(bool enabled, const std::string &server, int port,
               const std::string &topic, const std::string &username,
               const std::string &password, const std::string &client_id,
               const std::string &api_base_url,
               const std::string &local_path, int cooldown_seconds = 5);
  ~AlertService();

  // 动态更新 MQTT 与存储配置
  void UpdateConfig(bool enabled, const std::string &server, int port,
                    const std::string &topic, const std::string &username,
                    const std::string &password, const std::string &client_id,
                    const std::string &api_base_url,
                    const std::string &local_path);

  // 设置告警冷却秒数 (消抖)
  void SetCooldownSeconds(int seconds) { cooldown_seconds_ = seconds; }

  // 注册二次确认回调 (两级检测). 不注册时行为不变 —— 直接用传入的帧落盘.
  // 详见 ConfirmCallback 的说明.
  void SetConfirmCallback(ConfirmCallback cb) { confirm_callback_ = cb; }

  // 设置告警图片存储上限 (0 表示不限制)
  void SetMaxSizeMB(int mb) { max_size_mb_ = mb; }
  // 设置告警图片保留天数 (0 表示不过期)
  void SetRetentionDays(int days) { retention_days_ = days; }
  // 设置告警图片最大张数 (0 表示不限制)
  void SetMaxFiles(int files) { max_files_ = files; }

  // 通知当前模型的类别数 (3 / 4 / ...)，AlertService 会自动切换类别名与配色
  // profile 注意: 一旦调用过 SetClassNames() 注入 sidecar 类别名, 本接口会变成
  // no-op, 让 sidecar 优先
  void SetNumClasses(int n);

  // 直接注入类别名 (英文 + 中文), 来自 RKNN 模型同目录的 sidecar 文件
  // (.json/.labels/.txt/.names). 一旦调用, 后续 SetNumClasses() 自动让位,
  // 避免内置 profile 把 sidecar 覆盖回去.
  void SetClassNames(const std::vector<std::string> &en_names,
                     const std::vector<std::string> &cn_names);

  // 当前生效的 profile 对应的英文类名列表 (供 /api/status 透传给前端)
  const std::vector<std::string> &GetActiveClassNames() const;
  const std::vector<std::string> &GetActiveClassNamesCN() const;

  // 处理每一帧的推理结果，进行防抖与告警逻辑判断
  void ProcessFrameResult(const std::string &cam_id,
                          const std::vector<std::string> &tasks,
                          const cv::Mat &frame,
                          const std::vector<DetectResult> &results);

private:
  bool enabled_ = true;
  std::string server_ = "139.196.123.7";
  int port_ = 8492;
  std::string topic_ = "$data";
  std::string username_ = "";
  std::string password_ = "";
  std::string client_id_ = "box_001";
  std::string api_base_url_ = "http://127.0.0.1:8081";
  std::string local_storage_path_ = "./alerts";
  int cooldown_seconds_ = 5; // 告警冷却秒数（默认 5 秒）
  
  void* mqtt_client_ = nullptr;
  std::mutex mqtt_mutex_; // 保护 MQTT 客户端的互斥锁
  void ConnectMQTT();
  void DisconnectMQTT();

  // 告警图片存储上限 (MB)，0 表示不限制。默认 200MB
  int max_size_mb_ = 200;
  // 告警图片保留天数，0 表示不过期。默认 7 天
  int retention_days_ = 7;
  // 告警图片最大张数，0 表示不限制。默认 1000 张
  int max_files_ = 1000;

  // 消抖控制：记录每个 (cam_id + "_" + event_type) 的上一次告警时间戳
  std::map<std::string, std::chrono::steady_clock::time_point>
      last_alert_time_map_;
  std::mutex cooldown_mutex_;

  // 两级检测: 第二级(高清二次确认)回调, 由 main 注册
  ConfirmCallback confirm_callback_ = nullptr;
  // 清理操作互斥：保证多 worker 不会同时跑 CleanupAlerts
  std::mutex cleanup_mutex_;

  // 目标追踪状态 (防抖与运动检测)
  std::mutex tracking_mutex_;
  std::map<std::string, std::vector<TrackedObject>> camera_tracks_map_;
  int next_track_id_ = 1;

  // 告警专属后台线程池 (队列+并发处理)
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> task_queue_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_workers_ = false;

  // 内部函数：本地保存抓拍图片（画框后保存）
  void SaveSnapshot(const AlertEvent &event, const cv::Mat &frame,
                    const std::vector<DetectResult> &results);

  // 内部函数：通过 MQTT 推送告警事件 JSON 到云端
  void PushToCloud(const AlertEvent &event);

  // 内部函数：清理过期 / 超容的告警图片
  //   Pass 1: 删除早于 retention_days_ 天的文件
  //   Pass 2: 若总量仍超 max_size_mb_，按文件名字典序删最旧的直到不超
  void CleanupAlerts();
};
