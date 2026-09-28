#pragma once
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <opencv2/opencv.hpp>

// 视频帧回调函数，传入相机ID、权限列表和解码后的OpenCV Mat对象
using FrameCallback = std::function<void(const std::string& cam_id, const std::vector<std::string>& tasks, const std::string& model, const cv::Mat& frame)>;

class VideoManager {
public:
    VideoManager() = default;
    ~VideoManager() = default;

    // 初始化管理器
    bool Init();

    // 增加一路 RTSP 视频流，内部利用瑞芯微 MPP 进行硬件解码
    bool AddCamera(const std::string& cam_id, const std::string& rtsp_url, const std::vector<std::string>& tasks = {}, const std::string& model = "");

    // 移除指定的摄像头
    bool RemoveCamera(const std::string& cam_id);

    // 注册解码后的数据回调
    void RegisterFrameCallback(FrameCallback cb);
    
    // 获取某路摄像头的最新一帧（用于 Web 预览）
    static cv::Mat GetLatestFrame(const std::string& cam_id);

    // 单独抓一帧原始分辨率 (用于告警存图).
    // 常驻链路跑在 640x360 以省 CPU, 直接拿那张存图会糊; 但为了偶尔一次的存图
    // 让 8 路一直扛着全尺寸解码不划算, 所以只在告警发生时临时开一条管道抓一帧.
    // 告警有冷却 (默认 5 秒/相机+类别), 调用频率很低.
    // 抓不到 (相机忙 / 新建连接超时) 返回空 Mat, 调用方应退回用推理帧.
    static cv::Mat GrabFullFrame(const std::string& cam_id);
    
    // 启动/停止所有摄像头
    void StartAll();
    void StopAll();
    void Clear();

private:
    // TODO: 结合 FFmpeg 做 RTSP 拉流解包
    // TODO: 结合 MPP 进行 H264/H265 硬件解码
    // TODO: 结合 RGA 将解码后的 YUV/NV12 数据零拷贝转换为 RGB cv::Mat 格式
};
