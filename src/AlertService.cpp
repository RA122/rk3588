#include "AlertService.hpp"
#include "Profiler.hpp"
#include "VideoManager.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <utility>
#include <vector>
#include <MQTTClient.h>

namespace fs = std::filesystem;

// ===== 模型类别布局 Profile =====
// 不同 RKNN 模型可能训练时类别顺序不同，这里把所有支持的 layout 列出来，
// 程序启动后通过 SetNumClasses() 自动选取匹配的 profile。
//
// Profile 索引与模型 num_classes 一一对应：
//   num_classes = 4 -> profile_idx 0 (老 4 类模型)
//   num_classes = 3 -> profile_idx 1 (新 3 类模型，没有 fallen tree)
//
// 每个 profile 内部用 class_id(0..N-1) 做下标直接查表，O(1) 命中。

static const std::vector<std::vector<std::string>> g_class_names_profiles = {
    // ---------- Profile 0: 4 类模型 ----------
    {
        "fallen tree",   // Class 0: 倒伏树木
        "landslide",     // Class 1: 山体滑坡
        "road collapse", // Class 2: 道路塌陷
        "stone"          // Class 3: 碎石落石
    },
    // ---------- Profile 1: 3 类模型 ----------
    {
        "landslide",     // Class 0: 山体滑坡
        "road collapse", // Class 1: 道路塌陷
        "stone"          // Class 2: 碎石落石
    }};

static const std::vector<std::vector<std::string>> g_class_names_cn_profiles = {
    // ---------- Profile 0: 4 类 ----------
    {"倒伏树木", "山体滑坡", "道路塌陷", "碎石落石"},
    // ---------- Profile 1: 3 类 ----------
    {"山体滑坡", "道路塌陷", "碎石落石"}};

// 画框颜色已改为按类别名取 (见下面的 ColorForClassName), 不再按 class_id 索引,
// 所以这里不再维护"与 profile 一一对应"的颜色数组.

// 按类别名取画框颜色 (BGR). 色值与前端 _classStyle 的 Tailwind 色一一对应.
//
// 为什么不按 class_id 索引 g_color_profiles: 类别顺序取决于训练时的定义和 sidecar,
// 不同模型顺序不一样. 按下标取色会把颜色套到别的类别上 —— 比如石头拿到滑坡的颜色.
// 按名字取则无论顺序怎么变, 石头永远是紫色.
static cv::Scalar ColorForClassName(const std::string& name_in) {
    // 去首尾空白 + 转小写, 容忍 sidecar 里写成 "Stone" / " stone "
    std::string t = name_in;
    size_t b = t.find_first_not_of(" \t\r\n");
    size_t e = t.find_last_not_of(" \t\r\n");
    if (b == std::string::npos) return cv::Scalar(0, 255, 0);
    t = t.substr(b, e - b + 1);
    std::transform(t.begin(), t.end(), t.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    static const std::map<std::string, cv::Scalar> kColors = {
        {"stone",         cv::Scalar(246, 92, 139)},  // violet-500  RGB(139,92,246)  紫
        {"landslide",     cv::Scalar(36, 191, 251)},  // amber-500   RGB(251,191,36)  黄
        {"road collapse", cv::Scalar(94, 63, 244)},   // rose-500    RGB(244,63,94)   红
        {"fallen tree",   cv::Scalar(153, 211, 52)},  // emerald-500 RGB(52,211,153)  绿
    };
    auto it = kColors.find(t);
    if (it != kColors.end()) return it->second;
    return cv::Scalar(0, 255, 0);   // 未知类别兜底绿色
}

// 当前生效的 profile（默认 Profile 0 = 4 类，保持历史行为）
static int g_profile_idx = 0;

// 取当前 profile 的英文 / 中文表（指针指向 profile 中的元素，SetNumClasses
// 时切换）
static const std::vector<std::string> *g_class_names =
    &g_class_names_profiles[g_profile_idx];
static const std::vector<std::string> *g_class_names_cn =
    &g_class_names_cn_profiles[g_profile_idx];

// 来自模型 sidecar 的运行时类别名. 当 g_runtime_names_active=true 时,
// 这些向量接管 *g_class_names / *g_class_names_cn 指针, SetNumClasses 自动让位.
static std::vector<std::string> g_class_names_runtime;
static std::vector<std::string> g_class_names_cn_runtime;
static bool g_runtime_names_active = false;

static std::string GetClassName(int class_id) {
  if (class_id >= 0 && class_id < static_cast<int>(g_class_names->size())) {
    return (*g_class_names)[class_id];
  }
  return "class_" + std::to_string(class_id);
}

static std::string GetClassNameCN(int class_id) {
  if (class_id >= 0 && class_id < static_cast<int>(g_class_names_cn->size())) {
    return (*g_class_names_cn)[class_id];
  }
  return "未知告警";
}

AlertService::AlertService(bool enabled, const std::string &server, int port,
                           const std::string &topic,
                           const std::string &username,
                           const std::string &password,
                           const std::string &client_id,
                           const std::string &api_base_url,
                           const std::string &local_path, int cooldown_seconds)
    : enabled_(enabled), server_(server), port_(port), topic_(topic),
      username_(username), password_(password), client_id_(client_id),
      api_base_url_(api_base_url), local_storage_path_(local_path), cooldown_seconds_(cooldown_seconds) {
  stop_workers_ = false;
  int num_workers = 4;
  for (int i = 0; i < num_workers; ++i) {
    workers_.emplace_back([this]() {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queue_mutex_);
          this->condition_.wait(lock, [this]() {
            return this->stop_workers_ || !this->task_queue_.empty();
          });
          if (this->stop_workers_ && this->task_queue_.empty()) {
            return;
          }
          task = std::move(this->task_queue_.front());
          this->task_queue_.pop();
        }
        task();
      }
    });
  }

  if (!fs::exists(local_storage_path_)) {
    fs::create_directories(local_storage_path_);
  }
  std::cout << "[AlertService] Initialized. MQTT Enabled: "
            << (enabled_ ? "YES" : "NO") << ", Server: " << server_ << ":"
            << port_ << ", Topic: " << topic_ << ", ClientID: " << client_id_
            << ", API Base: " << api_base_url_
            << ", Storage: " << local_storage_path_
            << ", Cooldown: " << cooldown_seconds_ << "s" << std::endl;
            
  if (enabled_) {
    ConnectMQTT();
  }
}

const std::vector<std::string> &AlertService::GetActiveClassNames() const {
  return *g_class_names;
}

const std::vector<std::string> &AlertService::GetActiveClassNamesCN() const {
  return *g_class_names_cn;
}

void AlertService::SetNumClasses(int n) {
  // 已经被 sidecar 注入过 runtime 类别名 → 让位, 不再用内置 profile 覆盖
  if (g_runtime_names_active) {
    std::cout << "[AlertService] Runtime class names already active ("
              << g_class_names_runtime.size()
              << " classes), ignoring SetNumClasses(" << n
              << ") from legacy profile path" << std::endl;
    return;
  }

  int target = -1;
  switch (n) {
  case 4:
    target = 0;
    break; // Profile 0: 4 类 (老模型，含 fallen tree)
  case 3:
    target = 1;
    break; // Profile 1: 3 类 (新模型，无 fallen tree)
  default:
    std::cerr << "[AlertService] WARNING: unsupported num_classes=" << n
              << " (no sidecar either), keeping current profile "
              << g_profile_idx << " ("
              << g_class_names_profiles[g_profile_idx].size() << " classes)"
              << std::endl;
    return;
  }
  if (target == g_profile_idx)
    return;

  g_profile_idx = target;
  g_class_names = &g_class_names_profiles[g_profile_idx];
  g_class_names_cn = &g_class_names_cn_profiles[g_profile_idx];

  std::cout << "[AlertService] Switched to class profile " << g_profile_idx
            << " (" << g_class_names->size() << " classes):";
  for (const auto &s : *g_class_names)
    std::cout << " " << s;
  std::cout << std::endl;
}

void AlertService::SetClassNames(const std::vector<std::string> &en_names,
                                 const std::vector<std::string> &cn_names) {
  if (en_names.empty()) {
    std::cerr
        << "[AlertService] SetClassNames called with empty en_names, ignored"
        << std::endl;
    return;
  }
  g_class_names_runtime = en_names;
  g_class_names_cn_runtime =
      cn_names.size() == en_names.size() ? cn_names : en_names;
  g_runtime_names_active = true;

  // 切换指针到 runtime, 之后 GetClassName/GetClassNameCN/GetActiveClassNames
  // 都从这里走
  g_class_names = &g_class_names_runtime;
  g_class_names_cn = &g_class_names_cn_runtime;

  std::cout << "[AlertService] Installed runtime class names from sidecar ("
            << en_names.size() << " classes):";
  for (const auto &s : en_names)
    std::cout << " " << s;
  std::cout << std::endl;
}

void AlertService::UpdateConfig(bool enabled, const std::string &server,
                                int port, const std::string &topic,
                                const std::string &username,
                                const std::string &password,
                                const std::string &client_id,
                                const std::string &api_base_url,
                                const std::string &local_path) {
  enabled_ = enabled;
  server_ = server;
  port_ = port;
  topic_ = topic;
  username_ = username;
  password_ = password;
  client_id_ = client_id;
  api_base_url_ = api_base_url;
  local_storage_path_ = local_path;
  if (!fs::exists(local_storage_path_)) {
    fs::create_directories(local_storage_path_);
  }
  std::cout << "[AlertService] Config updated. MQTT Enabled: "
            << (enabled_ ? "YES" : "NO") << ", Server: " << server_ << ":"
            << port_ << ", Topic: " << topic_ << ", API Base: " << api_base_url_ << std::endl;
            
  DisconnectMQTT();
  if (enabled_) {
    ConnectMQTT();
  }
}

AlertService::~AlertService() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_workers_ = true;
  }
  condition_.notify_all();
  for (std::thread &worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  DisconnectMQTT();
  std::cout << "[AlertService] Destroyed." << std::endl;
}

void AlertService::ConnectMQTT() {
    std::lock_guard<std::mutex> lock(mqtt_mutex_);
    if (mqtt_client_ != nullptr) return; // Already connected

    std::string address = "tcp://" + server_ + ":" + std::to_string(port_);
    MQTTClient client;
    int rc = MQTTClient_create(&client, address.c_str(), client_id_.c_str(),
                               MQTTCLIENT_PERSISTENCE_NONE, NULL);
    if (rc != MQTTCLIENT_SUCCESS) {
        std::cerr << "[AlertService] Failed to create MQTT client, return code " << rc << std::endl;
        mqtt_client_ = nullptr;
        return;
    }
    
    mqtt_client_ = client;
    
    MQTTClient_connectOptions conn_opts = MQTTClient_connectOptions_initializer;
    conn_opts.keepAliveInterval = 20;
    conn_opts.cleansession = 1;
    if (!username_.empty()) {
        conn_opts.username = username_.c_str();
    }
    if (!password_.empty()) {
        conn_opts.password = password_.c_str();
    }
    
    rc = MQTTClient_connect(client, &conn_opts);
    if (rc != MQTTCLIENT_SUCCESS) {
        std::cerr << "[AlertService] Failed to connect to MQTT broker " << address 
                  << ", return code " << rc << std::endl;
        MQTTClient_destroy(&client);
        mqtt_client_ = nullptr;
    } else {
        std::cout << "[AlertService] Connected to MQTT broker " << address << std::endl;
    }
}

void AlertService::DisconnectMQTT() {
    std::lock_guard<std::mutex> lock(mqtt_mutex_);
    if (mqtt_client_ != nullptr) {
        MQTTClient client = static_cast<MQTTClient>(mqtt_client_);
        MQTTClient_disconnect(client, 1000);
        MQTTClient_destroy(&client);
        mqtt_client_ = nullptr;
        std::cout << "[AlertService] Disconnected from MQTT broker." << std::endl;
    }
}


void AlertService::ProcessFrameResult(
    const std::string &cam_id, const std::vector<std::string> &tasks,
    const cv::Mat &frame, const std::vector<DetectResult> &results) {
  if (results.empty())
    return;

  // 1. 基于配置权限过滤初步告警目标
  std::vector<DetectResult> valid_results;
  for (const auto &res : results) {
    std::string type_str = GetClassName(res.class_id);
    if (tasks.empty() ||
        std::find(tasks.begin(), tasks.end(), type_str) != tasks.end()) {
      valid_results.push_back(res);
    }
  }

  if (valid_results.empty())
    return;

  auto calc_iou = [](const cv::Rect &a, const cv::Rect &b) {
    int inter_area = (a & b).area();
    if (inter_area == 0)
      return 0.0f;
    int union_area = a.area() + b.area() - inter_area;
    return static_cast<float>(inter_area) / union_area;
  };

  // --- 质心追踪与运动防抖 (Centroid Tracker) ---
  std::vector<DetectResult> motion_valid_results;
  {
    std::lock_guard<std::mutex> track_lock(tracking_mutex_);
    auto &tracks = camera_tracks_map_[cam_id];

    std::vector<cv::Point2f> new_centroids;
    for (const auto &res : valid_results) {
      new_centroids.push_back(cv::Point2f(res.box.x + res.box.width / 2.0f,
                                          res.box.y + res.box.height / 2.0f));
    }

    std::vector<bool> matched_tracks(tracks.size(), false);
    std::vector<bool> matched_results(valid_results.size(), false);

    for (size_t i = 0; i < valid_results.size(); ++i) {
      int best_track_idx = -1;
      // 动态匹配阈值：收紧阈值防止不同目标来回闪烁导致的 ID
      // 切换。最大允许移动距离为长宽之和的 30%，保底 150 像素以适应落石的高速下落
      float min_dist = std::max(
          150.0f,
          (valid_results[i].box.width + valid_results[i].box.height) * 0.3f);

      for (size_t j = 0; j < tracks.size(); ++j) {
        if (matched_tracks[j] ||
            tracks[j].class_id != valid_results[i].class_id)
          continue;

        float dx = new_centroids[i].x - tracks[j].centroid.x;
        float dy = new_centroids[i].y - tracks[j].centroid.y;
        float dist = std::sqrt(dx * dx + dy * dy);

        if (dist < min_dist) {
          min_dist = dist;
          best_track_idx = static_cast<int>(j);
        }
      }

      if (best_track_idx >= 0) {
        matched_tracks[best_track_idx] = true;
        matched_results[i] = true;

        // 使用交并比 (IoU)
        // 判断是否真正发生了物理位移，能完美过滤手抖、呼吸效应及大目标边框跳变
        float iou =
            calc_iou(valid_results[i].box, tracks[best_track_idx].start_box);

        tracks[best_track_idx].centroid = new_centroids[i];
        tracks[best_track_idx].class_id =
            valid_results[i].class_id; // 允许目标在运动中改变类别（比如从 stone
                                       // 变成 landslide）
        tracks[best_track_idx].active_frames_count++;
        tracks[best_track_idx].lost_frames_count = 0;

        // 维护最近 15 帧的轨迹 (约 3 秒)
        tracks[best_track_idx].history_boxes.push_back(valid_results[i].box);
        if (tracks[best_track_idx].history_boxes.size() > 15) {
          tracks[best_track_idx].history_boxes.erase(
              tracks[best_track_idx].history_boxes.begin());
        }

        // 【核心修复】：检测目标是否重新归于静止
        // (比如手晃完图片后停在了新位置，或者落石滚到了山脚停下) 如果在过去的 3
        // 秒内，它几乎没怎么动（和 3 秒前位置的 IoU >
        // 0.75），那么重置它的“起始原点”
        if (tracks[best_track_idx].history_boxes.size() == 15) {
          float rest_iou =
              calc_iou(valid_results[i].box,
                       tracks[best_track_idx].history_boxes.front());
          if (rest_iou > 0.75f) {
            tracks[best_track_idx].start_box =
                valid_results[i].box; // 重置起跑线，停止无休止的告警
            tracks[best_track_idx].start_centroid = new_centroids[i]; // 同步重置初始质心
          }
        }

        // 计算质心的绝对物理像素位移
        float dx_start = new_centroids[i].x - tracks[best_track_idx].start_centroid.x;
        float dy_start = new_centroids[i].y - tracks[best_track_idx].start_centroid.y;
        float dist_start = std::sqrt(dx_start * dx_start + dy_start * dy_start);

        // 1. 动态滚落告警
        // 防抖策略：连续存活超过 5 帧，与初始位置的 IoU 下降到 0.5 以下，且绝对物理位移大于 30 像素
        if (tracks[best_track_idx].active_frames_count >= 5 && iou < 0.5f && dist_start > 30.0f) {
          motion_valid_results.push_back(valid_results[i]);
          tracks[best_track_idx].has_triggered_static_alarm = true;
        } 
        // 2. 静态落地/突现告警
        // 如果是一个新目标，并且稳定存在了 5 帧，即使它是静止的，我们也应该告警一次
        // 专门针对陡坡落石：掉下来很快（跟踪不到），落地后静止。只报一次，之后只要不动就不报。
        else if (tracks[best_track_idx].active_frames_count >= 5 && !tracks[best_track_idx].has_triggered_static_alarm) {
          motion_valid_results.push_back(valid_results[i]);
          tracks[best_track_idx].has_triggered_static_alarm = true; // 锁定，防止后续变成无休止的静止告警
        }
      }
    }

    // 创建新追踪对象
    for (size_t i = 0; i < valid_results.size(); ++i) {
      if (!matched_results[i]) {
        TrackedObject obj;
        obj.id = next_track_id_++;
        obj.class_id = valid_results[i].class_id;
        obj.centroid = new_centroids[i];
        obj.start_centroid = new_centroids[i];
        obj.start_box = valid_results[i].box;
        obj.history_boxes.push_back(valid_results[i].box);
        obj.active_frames_count = 1;
        obj.moving_frames_count = 0;
        obj.lost_frames_count = 0;
        tracks.push_back(obj);
      }
    }

    // 清理丢失过久的追踪对象
    std::vector<TrackedObject> active_tracks;
    for (size_t j = 0; j < tracks.size(); ++j) {
      // 如果 j 超出了 matched_tracks
      // 的原始长度，说明这是刚刚新加入的追踪对象，不需要增加 lost_frames_count
      bool is_matched = (j < matched_tracks.size()) ? matched_tracks[j] : true;

      if (!is_matched) {
        tracks[j].lost_frames_count++;
      }
      if (tracks[j].lost_frames_count < 10) { // 忍受 10 帧的丢失
        active_tracks.push_back(tracks[j]);
      }
    }
    tracks = active_tracks;
  }

  if (motion_valid_results.empty())
    return;

  float max_conf = 0.0f;
  std::vector<int> seen_class_ids;
  auto mark_seen = [&](int cid) {
    for (int v : seen_class_ids)
      if (v == cid)
        return false;
    seen_class_ids.push_back(cid);
    return true;
  };
  for (const auto &res : motion_valid_results) {
    if (res.confidence > max_conf) {
      max_conf = res.confidence;
    }
    mark_seen(res.class_id);
  }

  // 把所有命中类别写入 event_types 与 event_names_cn（用于多标签展示）
  std::vector<std::string> event_types;
  std::vector<std::string> event_names_cn;
  for (int cid : seen_class_ids) {
    event_types.push_back(GetClassName(cid));
    event_names_cn.push_back(GetClassNameCN(cid));
  }

  // 2. 告警消抖 (Cooldown) 校验 - 改为按 (cam + "+" + 类别签名)
  // 区分，降低多类别告警重复推送概率
  std::string type_signature;
  for (size_t i = 0; i < event_types.size(); ++i) {
    if (i)
      type_signature += "+";
    type_signature += event_types[i];
  }
  std::string key = cam_id + "@" + type_signature;
  auto now_steady = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(cooldown_mutex_);
    if (last_alert_time_map_.find(key) != last_alert_time_map_.end()) {
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                         now_steady - last_alert_time_map_[key])
                         .count();
      if (elapsed < cooldown_seconds_) {
        return;
      }
    }
    last_alert_time_map_[key] = now_steady;
  }

  // 覆盖 valid_results 以供后续告警推送使用
  valid_results = motion_valid_results;

  std::cout << "[AlertService] ALARM! Cam: " << cam_id << " - Detected "
            << valid_results.size() << " authorized disaster targets across "
            << event_types.size() << " class(es): " << type_signature
            << std::endl;

  // 构造告警事件结构体
  AlertEvent event;
  event.cam_id = cam_id;
  event.cam_name = "通道 " + cam_id;
  event.event_types = event_types;
  event.event_names_cn = event_names_cn;
  // 主类别（向后兼容）：取第一个类别
  event.event_type = event_types.front();
  event.event_name_cn = event_names_cn.front();
  event.max_confidence = max_conf;
  event.targets = valid_results;

  // 时间戳转换 (使用线程安全的 localtime_r)
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  struct tm tm_info;
  localtime_r(&in_time_t, &tm_info);

  char raw_buf[64];
  std::strftime(raw_buf, sizeof(raw_buf), "%Y%m%d_%H%M%S", &tm_info);
  event.raw_timestamp = raw_buf;

  char std_buf[64];
  std::strftime(std_buf, sizeof(std_buf), "%Y-%m-%d %H:%M:%S", &tm_info);
  event.timestamp = std_buf;

  // 文件名：用 + 拼接多类别，例如 D1_landslide+stone_20260917_154000.jpg
  std::string type_slug = type_signature;
  // 防御性清洗，避免极端类名包含特殊字符
  for (char &c : type_slug) {
    if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' ||
        c == '<' || c == '>' || c == '|') {
      c = '_';
    }
  }
  event.image_name =
      event.cam_id + "_" + type_slug + "_" + event.raw_timestamp + ".jpg";
  event.image_path = local_storage_path_ + "/" + event.image_name;

  // 3 & 4. 异步处理 IO
  // 任务：将告警存盘和推送任务丢入专属线程池，防止大规模并发时 CPU 毛刺
  cv::Mat frame_copy = frame.clone();
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    task_queue_.emplace([this, event, frame_copy, valid_results]() {
      // 两级检测第二级: 落盘前用高清帧再检一次, 过滤低分辨率下的误检.
      // 放在 IO worker 里做是因为抓帧要新建管道(0.5~2s), 不能堵住推理 worker.
      cv::Mat snap_frame = frame_copy;
      std::vector<DetectResult> snap_results = valid_results;

      if (this->confirm_callback_) {
        cv::Mat full;
        std::vector<DetectResult> full_results;
        if (this->confirm_callback_(event.cam_id, valid_results, full, full_results)) {
          snap_frame   = full;          // 高清帧
          snap_results = full_results;  // 高清帧上检出的框 (同一坐标系)
        } else {
          std::cout << "[AlertService] 二次确认未通过 (高清帧上无同类目标), "
                    << "判定误检已丢弃, cam=" << event.cam_id << std::endl;
          return;   // 不存图、不推送
        }
      }

      SaveSnapshot(event, snap_frame, snap_results);
      PushToCloud(event);
    });
  }
  condition_.notify_one();
}

void AlertService::SaveSnapshot(const AlertEvent &event, const cv::Mat &frame,
                                const std::vector<DetectResult> &results) {
  ScopedTimer t_total("Alert.SaveSnapshot");
  if (frame.empty())
    return;

  // frame 与 results 必定同坐标系: 走二次确认时是高清帧 + 高清帧上检出的框,
  // 没走确认时是原始推理帧 + 其检测框. 两种情况都直接画, 不需要坐标换算.
  cv::Mat canvas = frame.clone();

  for (const auto &res : results) {
    // 按类别名取色而不是按 class_id 索引, 详见 ColorForClassName 的说明
    cv::Scalar color = ColorForClassName(GetClassName(res.class_id));

    const cv::Rect& box = res.box;

    // 1. 绘制主矩形框 (线条加厚至 4px)
    cv::rectangle(canvas, box, color, 4);

    // 2. 构造动态类别与置信度文本 (如 fallen tree 88%)
    // 用 std::lround 做四舍五入而不是 static_cast<int> 截断，
    // 否则 0.456/0.499 都会显示成 45% / 49%，肉眼看上去就"全是 50%"
    std::string label_str =
        GetClassName(res.class_id) + " " +
        std::to_string(static_cast<int>(std::lround(res.confidence * 100.0f))) +
        "%";

    const double font_scale = 0.7;
    const int font_thick = 2;

    int baseline = 0;
    cv::Size text_size =
        cv::getTextSize(label_str, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thick, &baseline);

    // 3. 绘制标签顶部实心背景底色块
    int label_y = std::max(0, box.y - text_size.height - 8);
    cv::Rect label_rect(box.x, label_y, text_size.width + 10,
                        text_size.height + 8);
    cv::rectangle(canvas, label_rect, color, cv::FILLED);

    // 4. 绘制白色文字
    cv::putText(canvas, label_str,
                cv::Point(box.x + 5, label_y + text_size.height + 2),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), font_thick,
                cv::LINE_AA);
  }

  // 优化：降低 JPEG 质量到 80（OpenCV 默认是
  // 95），可以大幅降低编码时间和图片体积
  std::vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
  compression_params.push_back(80);
  bool ok = cv::imwrite(event.image_path, canvas, compression_params);
  if (ok) {
    std::cout << "[AlertService] Snapshot saved to: " << event.image_path
              << " (Drawn " << results.size() << " target boxes)." << std::endl;

    // 写盘成功后顺手清理：过期 + 超容（只在写盘成功的 worker 上触发，避免无效
    // IO）
    CleanupAlerts();
  } else {
    std::cerr << "[AlertService] Failed to save snapshot: " << event.image_path
              << std::endl;
  }
}

void AlertService::CleanupAlerts() {
  // 多 worker 同时写盘时，只让一个跑清理，其他直接
  // return，避免互相删对方刚写的文件
  std::lock_guard<std::mutex> lock(cleanup_mutex_);

  if (!fs::exists(local_storage_path_) ||
      !fs::is_directory(local_storage_path_)) {
    return;
  }

  int total_deleted = 0;

  // ===== Pass 1: 按 retention_days_ 删除过期文件 =====
  if (retention_days_ > 0) {
    try {
      auto now_file = fs::file_time_type::clock::now();
      auto threshold = now_file - std::chrono::hours(24LL * retention_days_);
      int deleted_by_age = 0;
      for (const auto &entry : fs::directory_iterator(local_storage_path_)) {
        if (entry.path().extension() != ".jpg")
          continue;
        try {
          auto ftime = fs::last_write_time(entry.path());
          if (ftime < threshold) {
            fs::remove(entry.path());
            deleted_by_age++;
          }
        } catch (...) { /* 单文件权限问题不影响整体 */
        }
      }
      if (deleted_by_age > 0) {
        std::cout << "[AlertService] Retention cleanup: deleted "
                  << deleted_by_age << " files older than " << retention_days_
                  << " days." << std::endl;
        total_deleted += deleted_by_age;
      }
    } catch (const std::exception &e) {
      std::cerr << "[AlertService] Retention cleanup error: " << e.what()
                << std::endl;
    }
  }

  // ===== Pass 2: 按 max_size_mb_ 或 max_files_ 删除最旧的，直到不超 =====
  if (max_size_mb_ > 0 || max_files_ > 0) {
    try {
      // 收集所有 .jpg 的 (路径, 大小)
      std::vector<std::pair<fs::path, std::uintmax_t>> files;
      std::uintmax_t total = 0;
      for (const auto &entry : fs::directory_iterator(local_storage_path_)) {
        if (entry.path().extension() != ".jpg")
          continue;
        try {
          auto size = fs::file_size(entry.path());
          files.emplace_back(entry.path(), size);
          total += size;
        } catch (...) {
        }
      }

      std::uintmax_t limit_bytes =
          static_cast<std::uintmax_t>(max_size_mb_) * 1024 * 1024;
      bool over_size = (max_size_mb_ > 0 && total > limit_bytes);
      bool over_count =
          (max_files_ > 0 && files.size() > static_cast<size_t>(max_files_));

      if ((over_size || over_count) && !files.empty()) {
        std::sort(files.begin(), files.end(), [](const auto &a, const auto &b) {
          return fs::last_write_time(a.first) < fs::last_write_time(b.first);
        });

        size_t current_count = files.size();
        int deleted_by_quota = 0;
        for (const auto &[path, size] : files) {
          bool size_exceeded = (max_size_mb_ > 0 && total > limit_bytes);
          bool count_exceeded =
              (max_files_ > 0 &&
               current_count > static_cast<size_t>(max_files_));

          if (!size_exceeded && !count_exceeded)
            break;

          try {
            fs::remove(path);
            total -= size;
            current_count--;
            deleted_by_quota++;
          } catch (...) {
          }
        }
        if (deleted_by_quota > 0) {
          std::cout << "[AlertService] Quota cleanup: deleted "
                    << deleted_by_quota
                    << " files. (Size limit: " << max_size_mb_
                    << "MB, Count limit: " << max_files_ << ")." << std::endl;
          total_deleted += deleted_by_quota;
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "[AlertService] Quota cleanup error: " << e.what()
                << std::endl;
    }
  }

  if (total_deleted > 0) {
    std::cout << "[AlertService] Cleanup finished, total deleted: "
              << total_deleted << std::endl;
  }
}

#include "json.hpp"
#include <fstream>

static std::string FileToBase64(const std::string &filepath) {
  ScopedTimer t_b64("Alert.FileToBase64");
  std::ifstream ifs(filepath, std::ios::binary);
  if (!ifs.is_open())
    return "";
  std::vector<uchar> buffer((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
  if (buffer.empty())
    return "";

  static const char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string ret;
  ret.reserve(((buffer.size() + 2) / 3) * 4);

  int val = 0, valb = -6;
  for (uchar c : buffer) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      ret.push_back(table[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6)
    ret.push_back(table[((val << 8) >> (valb + 8)) & 0x3F]);
  while (ret.size() % 4)
    ret.push_back('=');
  return ret;
}

void AlertService::PushToCloud(const AlertEvent &event) {
  ScopedTimer t_total("Alert.PushToCloud");
  if (!enabled_) {
    std::cout
        << "[AlertService] MQTT is disabled in config. Skipping cloud push."
        << std::endl;
    return;
  }

  std::cout << "[AlertService] Pushing alert event to MQTT Broker [" << server_
            << ":" << port_ << "] Topic [" << topic_ << "]" << std::endl;

  // 构建生产级分层 JSON 协议
  nlohmann::json root;

  // 1. Header 消息头
  root["header"] = {{"version", "1.0"},
                    {"client_id", client_id_},
                    {"msg_type", "ALARM_EVENT"},
                    {"send_time", event.timestamp}};

  // 2. Alarm Info 告警主体内容
  nlohmann::json alarm;
  alarm["cam_id"] = event.cam_id;
  alarm["cam_name"] = event.cam_name;
  alarm["event_type"] = event.event_type;
  alarm["event_name_cn"] = event.event_name_cn;
  // 多类别（新协议）：数组形式，便于云端直接按多标签分类
  nlohmann::json types_arr = nlohmann::json::array();
  for (const auto &t : event.event_types)
    types_arr.push_back(t);
  alarm["event_types"] = types_arr;
  nlohmann::json names_cn_arr = nlohmann::json::array();
  for (const auto &n : event.event_names_cn)
    names_cn_arr.push_back(n);
  alarm["event_names_cn"] = names_cn_arr;
  alarm["timestamp"] = event.timestamp;
  alarm["max_confidence"] = event.max_confidence;
  alarm["target_count"] = event.targets.size();

  // 抓拍图片 HTTP 访问 URL 与本地文件路径
  alarm["snapshot_url"] = api_base_url_ + "/alerts/" + event.image_name;
  alarm["snapshot_path"] = event.image_path;

  // 详细目标框数组
  nlohmann::json targets_json = nlohmann::json::array();
  for (const auto &target : event.targets) {
    nlohmann::json t;
    t["class_id"] = target.class_id;
    t["class_name"] = GetClassName(target.class_id);
    t["class_name_cn"] = GetClassNameCN(target.class_id);
    t["confidence"] = target.confidence;
    t["bbox"] = {target.box.x, target.box.y, target.box.width,
                 target.box.height};
    targets_json.push_back(t);
  }
  alarm["targets"] = targets_json;

  // 将图片 Base64 放置在最下方，方便顶部优先展示清晰的元数据信息
  std::string b64 = FileToBase64(event.image_path);
  if (!b64.empty()) {
    alarm["snapshot_base64"] = "data:image/jpeg;base64," + b64;
  } else {
    alarm["snapshot_base64"] = "";
  }

  root["alarm_info"] = alarm;

  std::string json_str = root.dump();
  std::cout << "  [MQTT Payload] JSON length: " << json_str.size() << " bytes."
            << std::endl;

  // 如果使用了原生客户端，直接推送
  {
      std::lock_guard<std::mutex> lock(mqtt_mutex_);
      if (mqtt_client_ != nullptr) {
          MQTTClient client = static_cast<MQTTClient>(mqtt_client_);
          MQTTClient_message pubmsg = MQTTClient_message_initializer;
          pubmsg.payload = (void*)json_str.c_str();
          pubmsg.payloadlen = (int)json_str.length();
          pubmsg.qos = 0;
          pubmsg.retained = 0;
          MQTTClient_deliveryToken token;
          
          int rc = MQTTClient_publishMessage(client, topic_.c_str(), &pubmsg, &token);
          if (rc != MQTTCLIENT_SUCCESS) {
              std::cerr << "[AlertService] Failed to publish message via MQTT, return code " << rc << std::endl;
              // 注意：此时我们持有 mqtt_mutex_。如果不解锁就调用 Connect/Disconnect 会死锁。
              // 因为重连机制比较复杂，这里简单的做法是打印错误。
              // 实际的自动重连应该交给 MQTTClient 自身的配置或专门的后台线程。
              // 对于这个 demo，我们在配置更改时才重连即可，或者我们可以用 try_lock 重连。
              // 为了避免死锁，直接跳过。
          }
      } else {
          std::cerr << "[AlertService] MQTT client not connected, skipping publish." << std::endl;
      }
  }
}
