#include "VideoManager.hpp"
#include "Profiler.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <thread>

#include <mutex>

// 常驻推理链路的工作分辨率.
// 相机主码流多是 1080p, 全尺寸下的 videoconvert(YUV→BGR) 加后面的灰度化/高斯模糊
// 是 CPU 大头 (实测 8 路就把 CPU 吃到 ~50%). 在转 BGR 之前先缩到 640x360,
// 像素量降到约 1/9, 而且 videoscale 走的是缩放而非色彩转换, 便宜得多.
// 告警存图不靠这条链路 —— 见 GrabFullFrame(), 存图时单独抓原分辨率.
static const int kInferWidth  = 640;
static const int kInferHeight = 360;

struct CameraContext {
  std::string cam_id;
  std::string rtsp_url;
  std::vector<std::string> tasks;
  std::string model;
  std::atomic<bool> is_running{false};
  std::shared_ptr<std::thread> decode_thread;

  cv::Mat latest_frame;
  std::mutex frame_mutex;

  // 异步推理支持：避免推理阻塞拉流线程导致卡顿/丢帧
  std::atomic<bool> infer_in_progress{false};
  std::shared_ptr<std::thread> infer_thread;

  // 运动触发推理：画面静止时跳过 NPU 推理，大幅降低 NPU 占用
  cv::Mat prev_gray;                // 上一帧灰度图（用于帧间差分）
  bool has_prev_gray = false;       // 是否已有上一帧
  int static_frames = 0;            // 连续静止帧数（用于自适应背景更新）
};

static std::map<std::string, std::shared_ptr<CameraContext>> g_cameras;
static FrameCallback g_frame_callback;

#include <cstdlib>

bool VideoManager::Init() {
  // 移除全局的 OPENCV_FFMPEG_CAPTURE_OPTIONS 设置。
  // 因为如果设置为 rtsp_transport;tcp，当 fallback 到 FFmpeg 去拉取 https (HLS) 视频流时，
  // FFmpeg 的 http/https demuxer 会因为找不到 rtsp_transport 选项而直接报错退出，导致拉流失败。
  // RTSP 的 TCP 模式已经在 GStreamer 管道中通过 protocols=tcp 保证，FFmpeg 仅作兜底。
  std::cout << "[VideoManager] Initialized." << std::endl;
  return true;
}

bool VideoManager::AddCamera(const std::string &cam_id,
                             const std::string &rtsp_url,
                             const std::vector<std::string> &tasks,
                             const std::string &model) {
  if (g_cameras.find(cam_id) != g_cameras.end()) {
    std::cerr << "[VideoManager] Camera " << cam_id << " already exists."
              << std::endl;
    return false;
  }
  auto ctx = std::make_shared<CameraContext>();
  ctx->cam_id = cam_id;
  ctx->rtsp_url = rtsp_url;
  ctx->tasks = tasks;
  ctx->model = model;

  g_cameras[cam_id] = ctx;
  std::cout << "[VideoManager] Added camera: " << cam_id << " (" << rtsp_url
            << ") with " << tasks.size()
            << " tasks and model: " << (model.empty() ? "none" : model) << "."
            << std::endl;
  return true;
}

bool VideoManager::RemoveCamera(const std::string &cam_id) {
  auto it = g_cameras.find(cam_id);
  if (it != g_cameras.end()) {
    if (it->second->is_running) {
      it->second->is_running = false;
      if (it->second->decode_thread && it->second->decode_thread->joinable()) {
        it->second->decode_thread->join();
      }
    }
    g_cameras.erase(it);
    std::cout << "[VideoManager] Removed camera: " << cam_id << std::endl;
    return true;
  }
  return false;
}

void VideoManager::RegisterFrameCallback(FrameCallback callback) {
  g_frame_callback = callback;
  std::cout << "[VideoManager] Frame callback registered." << std::endl;
}

cv::Mat VideoManager::GetLatestFrame(const std::string &cam_id) {
  auto it = g_cameras.find(cam_id);
  if (it != g_cameras.end()) {
    std::lock_guard<std::mutex> lock(it->second->frame_mutex);
    return it->second->latest_frame.clone();
  }
  return cv::Mat();
}

cv::Mat VideoManager::GrabFullFrame(const std::string &cam_id) {
  auto it = g_cameras.find(cam_id);
  if (it == g_cameras.end())
    return cv::Mat();
  auto ctx = it->second;

  // 与常驻管道的区别: 没有 videorate 节流, 也没有 videoscale 降采样, 要的就是原始清晰度.
  // latency 给到 300ms, 新建连接时容忍更大的抖动.
  std::string pipeline;
  if (ctx->rtsp_url.find("rtsp://") == 0 || ctx->rtsp_url.find("rtsps://") == 0) {
    pipeline = "rtspsrc location=\"" + ctx->rtsp_url +
               "\" protocols=tcp latency=300 ! "
               "decodebin ! "
               "videoconvert ! video/x-raw,format=BGR ! "
               "appsink max-buffers=1 drop=true sync=false";
  } else {
    pipeline = "uridecodebin uri=\"" + ctx->rtsp_url + "\" ! "
               "videoconvert ! video/x-raw,format=BGR ! "
               "appsink max-buffers=1 drop=true sync=false";
  }

  cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
  if (!cap.isOpened()) {
    std::cerr << "[VideoManager] GrabFullFrame: cannot open pipeline for "
              << cam_id << std::endl;
    return cv::Mat();
  }

  // 新起的 RTSP 连接要等关键帧才出图, 给 ~2 秒的尝试窗口
  cv::Mat frame;
  for (int i = 0; i < 20; ++i) {
    if (cap.grab() && cap.retrieve(frame) && !frame.empty())
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  cap.release();

  if (frame.empty()) {
    std::cerr << "[VideoManager] GrabFullFrame: got no frame for " << cam_id
              << std::endl;
  }
  return frame;
}

void VideoManager::StartAll() {
  int start_delay_ms = 0;
  for (auto &pair : g_cameras) {
    auto ctx = pair.second;
    if (!ctx->is_running) {
      ctx->is_running = true;
      int current_delay = start_delay_ms;
      start_delay_ms += 500; // 每路相机错开 500ms 启动，避免并发初始化失败

      ctx->decode_thread = std::make_shared<std::thread>([ctx,
                                                          current_delay]() {
        if (current_delay > 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(current_delay));
        }

        // ====== 统一 GStreamer Pipeline ======
        // decodebin 会自动识别 H.264/H.265，并自动调用底层的 mppvideodec
        // 进行硬解。 rtspsrc 自带 jitterbuffer（由latency=100控制），无需再额外串联 rtpjitterbuffer。
        // rtspsrc 自身也支持原生解析 rtsp://user:pass@host 格式，无需手动拆分。
        std::string pipeline_gstreamer;
        if (ctx->rtsp_url.find("rtsp://") == 0 || ctx->rtsp_url.find("rtsps://") == 0) {
            // 注意顺序: 先 videoscale 再 videoconvert.
            // 反过来 (先转 BGR 再缩放) 等于让最贵的那步在全分辨率上做, 白搭.
            pipeline_gstreamer =
                "rtspsrc location=\"" + ctx->rtsp_url +
                "\" protocols=tcp latency=100 ! "
                "decodebin ! "
                "videorate ! video/x-raw,framerate=5/1 ! "
                "videoscale ! video/x-raw,width=" + std::to_string(kInferWidth) +
                ",height=" + std::to_string(kInferHeight) + " ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink max-buffers=1 drop=true sync=false";
        } else {
            // HLS, HTTP, RTMP 等其他协议使用万能的 uridecodebin
            pipeline_gstreamer =
                "uridecodebin uri=\"" + ctx->rtsp_url + "\" ! "
                "videorate ! video/x-raw,framerate=5/1 ! "
                "videoscale ! video/x-raw,width=" + std::to_string(kInferWidth) +
                ",height=" + std::to_string(kInferHeight) + " ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink max-buffers=1 drop=true sync=false";
        }

        cv::VideoCapture cap(pipeline_gstreamer, cv::CAP_GSTREAMER);

        if (!cap.isOpened()) {
          // ====== Pipeline 2: FFmpeg 兜底（纯软解，性能最差但兼容性最好）
          // ======
          std::cerr
              << "[VideoManager] GStreamer failed, falling back to FFmpeg: "
              << ctx->cam_id << std::endl;
          cap.open(ctx->rtsp_url, cv::CAP_FFMPEG);
          if (!cap.isOpened()) {
            std::cerr << "[VideoManager] Failed to open stream: "
                      << ctx->rtsp_url << std::endl;
            return;
          }
        }

        cv::Mat bgr_frame;
        while (ctx->is_running) {
          // 每 200ms 读取一帧 (5 FPS)
          // 由于 appsink drop=false，延时会让 GStreamer 管道阻塞，
          // 触发 queue leaky=downstream 丢掉多余的帧
          std::this_thread::sleep_for(std::chrono::milliseconds(200));

          ScopedTimer timer_loop("Video.FrameLoop");

          if (cap.grab()) {
            ScopedTimer t_grab("Video.grab");
            if (cap.retrieve(bgr_frame)) {
              // 1. 更新最新帧：用 clone 替换 copyTo，避免读写竞争花屏
              {
                ScopedTimer t_copy("Video.CopyToLatest");
                std::lock_guard<std::mutex> lock(ctx->frame_mutex);
                ctx->latest_frame = bgr_frame.clone();
              }

              // 2. 运动触发推理：画面静止时跳过 NPU 推理
              //    - 转灰度 + 高斯模糊去噪
              //    - 帧间差分 + 阈值化
              //    - 统计变化像素数，低于阈值则判定为静止
              //    - 静止时缓慢更新背景（适应白天黑夜光线渐变）
              bool should_infer = true;
              if (ctx->model.empty()) {
                // 该通道没绑模型 → 推理压根不会发生, 运动检测算出来的结论没人用.
                // 直接跳过, 省掉每帧一次灰度化 + 5x5 高斯模糊.
                should_infer = false;
              } else {
                cv::Mat curr_gray;
                cv::cvtColor(bgr_frame, curr_gray, cv::COLOR_BGR2GRAY);
                cv::GaussianBlur(curr_gray, curr_gray, cv::Size(5, 5), 0);

                if (ctx->has_prev_gray) {
                  cv::Mat diff;
                  cv::absdiff(ctx->prev_gray, curr_gray, diff);
                  cv::threshold(diff, diff, 25, 255, cv::THRESH_BINARY);

                  int motion_pixels = cv::countNonZero(diff);
                  const int kMinMotionPixels = 500;  // 变化像素阈值

                  if (motion_pixels < kMinMotionPixels) {
                    // 判定为静止：不推理
                    should_infer = false;
                    ctx->static_frames++;
                    // 静止超过 3 帧后，背景慢慢向当前帧靠拢（适应光线渐变）
                    if (ctx->static_frames > 3) {
                      cv::addWeighted(ctx->prev_gray, 0.95, curr_gray, 0.05, 0,
                                      ctx->prev_gray);
                    }
                  } else {
                    ctx->static_frames = 0;
                  }
                }
                // 更新上一帧（静止时已更新背景，这里只在有运动时覆盖）
                if (should_infer) {
                  ctx->prev_gray = curr_gray.clone();
                  ctx->has_prev_gray = true;
                }
              }

              // 3. 异步推理：仅在画面有运动时触发
              //    上一个推理未完成时直接跳过这帧
              //    避免推理阻塞拉流线程 → 卡顿
              //    避免推理堆帧 → 告警延迟
              if (should_infer && !ctx->infer_in_progress.exchange(true)) {
                // 释放上一个推理线程（已结束）
                if (ctx->infer_thread && ctx->infer_thread->joinable()) {
                  ctx->infer_thread->join();
                }
                // 启动新推理线程（拷贝帧独立持有数据）
                ctx->infer_thread = std::make_shared<std::thread>(
                    [ctx, frame = bgr_frame.clone()]() {
                      if (g_frame_callback) {
                        ScopedTimer t_cb("Video.Callback");
                        g_frame_callback(ctx->cam_id, ctx->tasks, ctx->model,
                                         frame);
                      }
                      ctx->infer_in_progress = false;
                    });
              }
            }
          } else {
            std::cerr << "[VideoManager] Stream read error for " << ctx->cam_id
                      << std::endl;
            break;
          }
        }

        // 退出前等最后一个推理完成
        if (ctx->infer_thread && ctx->infer_thread->joinable()) {
          ctx->infer_thread->join();
        }
        cap.release();
      });
    }
  }
}

void VideoManager::StopAll() {
  std::cout << "[VideoManager] Stopping all streams..." << std::endl;
  for (auto &pair : g_cameras) {
    if (pair.second->is_running) {
      pair.second->is_running = false;
      if (pair.second->decode_thread &&
          pair.second->decode_thread->joinable()) {
        pair.second->decode_thread->join();
      }
    }
  }
}

void VideoManager::Clear() {
  StopAll();
  g_cameras.clear();
  std::cout << "[VideoManager] Cleared all cameras." << std::endl;
}
