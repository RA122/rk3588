#include "AlertService.hpp"
#include "Profiler.hpp"
#include "RKNNInferEngine.hpp"
#include "VideoManager.hpp"
#include "WebServer.hpp"
#include "json.hpp"
#include <algorithm>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>

using json = nlohmann::json;

// 推理任务结构体
struct InferTask {
  std::string cam_id;
  std::vector<std::string> tasks;
  std::string model;
  cv::Mat frame;
};

// 两级检测第二级(高清二次确认)用的置信度阈值.
// 可在 config.json 的 system.confirm_threshold 调整, 热重载即生效.
//
// 这个值不要设得比第一级(0.45)高太多: 第二级送进模型时, 目标尺度跟第一级基本一样
// (都是等比 letterbox 到模型的 640x640 输入), 所以置信度分布差别不大 ——
// 阈值一抬高, 真目标会跟着误检一起被滤掉.
//
// 二次确认能滤掉误检, 靠的是"高清下杂物更像杂物、自然检不出", 而不是靠抬阈值.
static float g_confirm_threshold = 0.40f;

// 辅助解析函数：兼容 boolean 与字符串 "true"/"false"
static bool ParseBool(const nlohmann::json &j, const std::string &key,
                      bool default_val) {
  if (!j.is_object() || !j.contains(key))
    return default_val;
  auto val = j[key];
  if (val.is_boolean())
    return val.get<bool>();
  if (val.is_string()) {
    std::string s = val.get<std::string>();
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
    return (s == "true" || s == "1");
  }
  if (val.is_number())
    return val.get<int>() != 0;
  return default_val;
}

// AI 算法盒子主程序骨架
int main(int argc, char **argv) {
  std::cout << "Starting AI Algorithm Box (Rockfall/Landslide/Road Collapse)..."
            << std::endl;

  std::string config_path = (argc > 1) ? argv[1] : "../config/config.json";
  std::string web_root = (argc > 2) ? argv[2] : "../web";

  // 默认初始值 (MQTT默认地址: 139.196.123.7:8492, 默认主题: $data)
  bool mqtt_enabled = true;
  std::string mqtt_server = "139.196.123.7";
  int mqtt_port = 8492;
  std::string mqtt_topic = "$data";
  std::string mqtt_user = "";
  std::string mqtt_pass = "";
  std::string client_id = "box_001";
  std::string api_base_url = "http://127.0.0.1:8081";

  std::string local_path = "./alerts";

  // 读取初次配置
  try {
    std::ifstream ifs(config_path);
    if (ifs.is_open()) {
      json cj = json::parse(ifs);
      if (cj.contains("mqtt")) {
        mqtt_enabled = ParseBool(cj["mqtt"], "enabled", mqtt_enabled);
        mqtt_server = cj["mqtt"].value("server", mqtt_server);
        mqtt_port = cj["mqtt"].value("port", mqtt_port);
        mqtt_topic = cj["mqtt"].value("topic", mqtt_topic);
        mqtt_user = cj["mqtt"].value("username", mqtt_user);
        mqtt_pass = cj["mqtt"].value("password", mqtt_pass);
        client_id = cj["mqtt"].value("client_id", client_id);
      }
      if (cj.contains("storage")) {
        local_path = cj["storage"].value("local_path", local_path);
      }
      if (cj.contains("system")) {
        api_base_url = cj["system"].value("api_base_url", api_base_url);
        g_confirm_threshold = cj["system"].value("confirm_threshold", g_confirm_threshold);
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] Initial config parsing failed: " << e.what() << std::endl;
  }

  // 保存告警清理上限的初始值 (供 AlertService 构造后 set)
  int initial_max_size_mb = 200;
  int initial_retention_days = 7;
  int initial_max_files = 1000;
  try {
    std::ifstream ifs2(config_path);
    if (ifs2.is_open()) {
      json cj2 = json::parse(ifs2);
      if (cj2.contains("storage")) {
        initial_max_size_mb =
            cj2["storage"].value("max_size_mb", initial_max_size_mb);
        initial_retention_days =
            cj2["storage"].value("retention_days", initial_retention_days);
        initial_max_files =
            cj2["storage"].value("max_files", initial_max_files);
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] Storage config parsing failed: " << e.what() << std::endl;
  }

  // 1.1 启动管理界面后台 Web 服务 (端口 8081)
  WebServer web_server(config_path, web_root);
  web_server.Start(8081);

  // 1.2 启动 Profiler 后台定期打印 (每 5 秒汇总一次各节点耗时)
  Profiler::Instance().StartAutoReport(std::chrono::seconds(5));

  // 2. 初始化核心 AI 组件
  VideoManager video_mgr;
  video_mgr.Init();
  AlertService alert_service(mqtt_enabled, mqtt_server, mqtt_port, mqtt_topic,
                             mqtt_user, mqtt_pass, client_id, api_base_url, local_path);

  // 应用 config.json 中的告警图片清理上限 (热重载路径同样会更新)
  alert_service.SetMaxSizeMB(initial_max_size_mb);
  alert_service.SetRetentionDays(initial_retention_days);
  alert_service.SetMaxFiles(initial_max_files);
  std::cout << "[System] Alert cleanup policy: max_size_mb="
            << initial_max_size_mb
            << ", retention_days=" << initial_retention_days << std::endl;

  // 多模型引擎池 (Key 变为 cam_id)
  std::map<std::string, std::shared_ptr<RKNNInferEngine>> engine_pool;
  std::mutex engine_pool_mutex;

  // 异步推理队列与线程池
  std::queue<InferTask> infer_queue;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::atomic<bool> is_running(true);

  // 推理工作线程
  std::vector<std::thread> infer_threads;
  for (int i = 0; i < 3; ++i) {
    infer_threads.emplace_back([&]() {
      while (is_running) {
        ScopedTimer t_worker("Worker.Total");
        InferTask task;
        {
          // 注意：cv::wait_for 必须用 unique_lock，它在等待时会自动 unlock，
          //       唤醒后再 lock。lock_guard 没有 unlock 接口，不能用于 cv。
          std::unique_lock<std::mutex> lock(queue_mutex);
          if (infer_queue.empty()) {
            queue_cv.wait_for(lock, std::chrono::milliseconds(50), [&] {
              return !infer_queue.empty() || !is_running;
            });
            if (!is_running && infer_queue.empty())
              break;
            if (infer_queue.empty())
              continue;
          }
          task = infer_queue.front();
          infer_queue.pop();
        }

        std::vector<DetectResult> results;
        if (!task.model.empty()) {
          std::shared_ptr<RKNNInferEngine> engine;
          {
            std::lock_guard<std::mutex> lock(engine_pool_mutex);
            if (engine_pool.find(task.cam_id) != engine_pool.end()) {
              engine = engine_pool[task.cam_id];
            }
          }
          if (engine) {
            results = engine->Inference(task.frame);
          }
        }

        {
          ScopedTimer t_alert("Worker.AlertProcess");
          alert_service.ProcessFrameResult(task.cam_id, task.tasks, task.frame,
                                           results);
        }
      }
    });
  }

  // 3. 注册解码帧回调函数
  video_mgr.RegisterFrameCallback(
      [&](const std::string &cam_id, const std::vector<std::string> &tasks,
          const std::string &model, const cv::Mat &frame) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (infer_queue.size() > 10) {
          infer_queue.pop();
        }
        infer_queue.push({cam_id, tasks, model, frame});
        queue_cv.notify_one();
      });

  // 3.1 注册两级检测的"二次确认"回调.
  // 常驻推理跑在降采样帧上(省 CPU), 杂物在低分辨率下容易和目标混淆 —— 比如桌前的
  // 黑色支架头缩到 360p 后跟石头几乎一样, 第一级会以 0.50 这种压线分数放过它.
  // 所以告警真正落盘前, 临时抓一帧高清图再检一次; 高清帧上检不出同类目标就判定
  // 误检直接丢弃. 确认通过时返回的帧与框来自同一张高清图, 存图无需坐标换算.
  alert_service.SetConfirmCallback([&engine_pool, &engine_pool_mutex](
                                       const std::string& cam_id,
                                       const std::vector<DetectResult>& low_results,
                                       cv::Mat& out_full_frame,
                                       std::vector<DetectResult>& out_results) -> bool {
      std::shared_ptr<RKNNInferEngine> engine;
      {
        std::lock_guard<std::mutex> lock(engine_pool_mutex);
        auto it = engine_pool.find(cam_id);
        if (it != engine_pool.end()) engine = it->second;
      }
      if (!engine) return false;   // 该通道没绑模型, 无从确认

      cv::Mat full = VideoManager::GrabFullFrame(cam_id);
      if (full.empty()) return false;   // 抓不到高清帧, 保守起见不算确认通过

      std::vector<DetectResult> res = engine->Inference(full);

      // 只认与第一级检出类别相同的目标
      std::vector<int> wanted;
      for (const auto& r : low_results) wanted.push_back(r.class_id);

      std::vector<DetectResult> confirmed;
      float best_same_class = 0.0f;
      for (const auto& r : res) {
          if (std::find(wanted.begin(), wanted.end(), r.class_id) == wanted.end()) continue;
          best_same_class = std::max(best_same_class, r.confidence);
          if (r.confidence < g_confirm_threshold) continue;
          confirmed.push_back(r);
      }

      // 诊断日志: 现场调阈值全靠它 —— 能看到高清帧上到底检出了什么、置信度多少
      std::cout << "[Confirm] cam=" << cam_id << " 高清帧 " << full.cols << "x" << full.rows
                << " | 共检出 " << res.size()
                << " 个, 同类最高置信度 " << best_same_class
                << " (阈值 " << g_confirm_threshold << ")"
                << ", 通过 " << confirmed.size() << " 个" << std::endl;

      if (confirmed.empty()) return false;

      out_full_frame = full;
      out_results    = confirmed;
      return true;
  });

  // 4. 定义重载配置的逻辑
  auto reload_config_func = [&]() {
    static std::mutex reload_mutex;
    std::lock_guard<std::mutex> reload_lock(reload_mutex);

    std::cout << "[System] Reloading configuration and cameras..." << std::endl;
    video_mgr.Clear();

    try {
      std::ifstream ifs(config_path);
      if (ifs.is_open()) {
        json config_json = json::parse(ifs);

        if (config_json.contains("mqtt")) {
          bool en = ParseBool(config_json["mqtt"], "enabled", true);
          std::string srv =
              config_json["mqtt"].value("server", "139.196.123.7");
          int port = config_json["mqtt"].value("port", 8492);
          std::string top = config_json["mqtt"].value("topic", "$data");
          std::string usr = config_json["mqtt"].value("username", "");
          std::string pwd = config_json["mqtt"].value("password", "");
          std::string cid = config_json["mqtt"].value("client_id", "box_001");
          std::string base_url = config_json.contains("system") ? config_json["system"].value("api_base_url", "http://127.0.0.1:8081") : "http://127.0.0.1:8081";
          if (config_json.contains("system")) {
            g_confirm_threshold = config_json["system"].value("confirm_threshold", g_confirm_threshold);
            std::cout << "[System] Confirm threshold = " << g_confirm_threshold << std::endl;
          }
          std::string lp =
              config_json.contains("storage")
                  ? config_json["storage"].value("local_path", "./alerts")
                  : "./alerts";
          alert_service.UpdateConfig(en, srv, port, top, usr, pwd, cid, base_url, lp);

          // 热重载时同步告警图片清理上限
          if (config_json.contains("storage")) {
            int new_max_mb = config_json["storage"].value("max_size_mb", 200);
            int new_ret_d = config_json["storage"].value("retention_days", 7);
            int new_max_files = config_json["storage"].value("max_files", 1000);
            alert_service.SetMaxSizeMB(new_max_mb);
            alert_service.SetRetentionDays(new_ret_d);
            alert_service.SetMaxFiles(new_max_files);
            std::cout << "[System] Alert cleanup policy updated: max_size_mb="
                      << new_max_mb << ", retention_days=" << new_ret_d
                      << ", max_files=" << new_max_files << std::endl;
          }
        }

        std::lock_guard<std::mutex> lock(engine_pool_mutex);
        engine_pool.clear(); // 清空旧引擎

        if (config_json.contains("cameras") &&
            config_json["cameras"].is_array()) {
          for (const auto &cam : config_json["cameras"]) {
            std::string id = cam["id"];
            std::string url = cam["url"];
            std::vector<std::string> tasks;
            if (cam.contains("tasks") && cam["tasks"].is_array()) {
              for (const auto &t : cam["tasks"]) {
                tasks.push_back(t.get<std::string>());
              }
            }
            std::string model = cam.value("model", "");

            if (!model.empty()) {
              std::cout
                  << "[System] Instantiating dedicated AI engine for camera: "
                  << id << " (" << model << ")" << std::endl;
              auto new_engine = std::make_shared<RKNNInferEngine>();
              std::string full_model_path = "../models/" + model;
              if (!new_engine->LoadModel(full_model_path)) {
                std::cerr << "WARNING: Failed to load RKNN model: " << model
                          << std::endl;
              }
              engine_pool[id] = new_engine;

              // 类别布局自适应：优先用 sidecar 里的运行时类别名, 找不到再退回
              // num_classes 选内置 profile
              //   sidecar:  RKNNInferEngine 自动找
              //   <model>.json/.labels/.txt/.names
              //             找到就调 SetClassNames 覆盖内置 profile
              //             (任意类别数都行)
              //   fallback: SetNumClasses, 但此时若 runtime 已激活, 内部会
              //   no-op
              const auto &en_names = new_engine->class_names_en();
              const auto &cn_names = new_engine->class_names_cn();
              if (!en_names.empty()) {
                alert_service.SetClassNames(en_names, cn_names);
              }

              // 类别布局自适应：直接从模型输出张量的形状推 num_classes，
              //   单输出 dim2 = 4 + num_classes (无 DFL) 或 64 + num_classes
              //   (有 DFL) 多输出 任一特征图通道数同样按 c - 4 / c - 64 推
              //   推理时 PostProcess 也会再次校验 / 覆盖，这里只是让
              //   AlertService 立刻拿到初始值。
              int nc = new_engine->num_classes();
              if (nc > 0) {
                alert_service.SetNumClasses(nc);
              }
            }

            video_mgr.AddCamera(id, url, tasks, model);
          }
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] Failed to parse config.json: " << e.what()
                << std::endl;
    }
    video_mgr.StartAll();
  };

  // 绑定热重载回调
  web_server.SetOnConfigUpdated(reload_config_func);
  
  // 绑定重启回调
  web_server.SetOnRestartEngine([&is_running]() {
    is_running = false;
  });

  // 把当前生效的模型类别名透传给 WebServer，前端 /api/status 会自动带上
  // model_classes 字段
  web_server.SetClassNamesProvider(
      [&alert_service]() { return alert_service.GetActiveClassNames(); },
      [&alert_service]() { return alert_service.GetActiveClassNamesCN(); });

  // 5. 启动系统
  std::cout << "System initialized. Starting video streams..." << std::endl;
  reload_config_func();

  // 阻塞主线程以保持服务运行
  while (is_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // 退出清理逻辑
  std::cout << "[System] Shutting down, stopping video streams..." << std::endl;
  video_mgr.StopAll();
  
  std::cout << "[System] Stopping inference workers..." << std::endl;
  is_running = false;
  queue_cv.notify_all();
  for (auto &th : infer_threads) {
    if (th.joinable())
      th.join();
  }
  
  web_server.Stop();
  std::cout << "[System] Clean exit complete." << std::endl;

  return 0;
}
