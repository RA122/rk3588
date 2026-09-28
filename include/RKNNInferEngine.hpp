#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include <rga/im2d.h>
#include <rga/rga.h>

// 检测目标结构体
struct DetectResult {
    int class_id;        // 类别ID（如：0-落石，1-凹陷）
    float confidence;    // 置信度
    cv::Rect box;        // 目标框边界
};

class RKNNInferEngine {
public:
    RKNNInferEngine();
    ~RKNNInferEngine();

    // 加载转换好的 RKNN 模型 (.rknn)
    bool LoadModel(const std::string& model_path);

    // 执行推理，传入图像，返回检测框列表
    std::vector<DetectResult> Inference(const cv::Mat& frame);

    // 当前模型类别数（首跑推理后才会填充；0 表示尚未推理）
    int num_classes() const { return num_classes_; }

    // 从 sidecar 读出来的英文 / 中文类别名. 空表示没找到 sidecar,
    // 此时由 AlertService 走兼容路径.
    const std::vector<std::string>& class_names_en() const { return class_names_en_; }
    const std::vector<std::string>& class_names_cn() const { return class_names_cn_; }

    // 公有静态 helper: 无需构造 engine 实例就能读任意 .rknn 的 sidecar 类别名.
    // WebServer 用这个接口响应 /api/model_classes 拉取任意模型的类别
    // (前端按模型绑定相机时自动填充 Tasks).
    // num_classes <= 0 表示不校验行数, 原样返回 sidecar 里的所有类.
    static bool ReadSidecarClasses(const std::string& model_path,
                                    int num_classes,
                                    std::vector<std::string>& out_en,
                                    std::vector<std::string>& out_cn);

private:
    // 实际底层变量：
    rknn_context ctx_;
    bool is_init_;

    // 输入输出属性
    rknn_tensor_attr* input_attrs_;
    rknn_tensor_attr* output_attrs_;
    rknn_input_output_num io_num_;
    
    // NPU 零拷贝输入内存
    rknn_tensor_mem* input_mem_ = nullptr;

    int model_width_;
    int model_height_;
    int model_channel_;

    // 当前模型实际输出的类别数（用于适配 3 类 / 4 类等不同布局）
    int num_classes_ = 0;

    // 从 sidecar 加载的类别名 (英文 / 中文). 留空表示没有 sidecar,
    // AlertService 会沿用内置 profile; sidecar 存在时, 这里就是权威来源.
    std::vector<std::string> class_names_en_;
    std::vector<std::string> class_names_cn_;

    // 后处理自适应：判断模型输出是否已包含 sigmoid 激活
    bool cls_activation_checked_ = false;
    // 模型类型标记
    bool is_dfl_ = false;
    bool is_transposed_ = false;
    bool model_has_sigmoid_ = false;

    // 防止同一个 engine (同一个摄像头) 被多个 Worker 线程同时执行 rknn_run 导致底层崩溃
    std::mutex infer_mutex_;

    // 辅助后处理函数
    void PostProcessYOLOv11(rknn_output* outputs, std::vector<DetectResult>& results);
    void PostProcessMultiOutput(rknn_output* outputs, std::vector<DetectResult>& results);
    void NMS(std::vector<DetectResult>& boxes, float nms_thresh);
    float CalculateIoU(const cv::Rect& a, const cv::Rect& b);

    // 尝试从模型同目录的 sidecar 文件里读取类别名 (支持 .json / .labels / .txt / .names).
    // 读取成功会填充 class_names_en_ / class_names_cn_, 返回 true;
    // 找不到任何 sidecar 返回 false, 调用方应当走兼容 profile 路径.
    bool LoadClassNamesFromSidecar(const std::string& model_path, int num_classes);

    // 从 output_attrs[] 反推 YOLO 类别数. 单输出 layout 优先; 多输出找第一个 nc∈[1,100] 的.
    static int DetectNumClassesFromOutputs(const rknn_tensor_attr* attrs, uint32_t n_output);
};
