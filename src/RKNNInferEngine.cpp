#include "RKNNInferEngine.hpp"
#include "Profiler.hpp"
#include "json.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <atomic>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

// ========== NPU 核心调度配置 ==========
// 默认启用显式的 Round-Robin 调度:
// 如果不指定 RKNN_NPU_CORE，每次初始化模型时自动轮询分配到 CORE_0, CORE_1,
// CORE_2。 这是因为如果统一设为 0_1_2 或 AUTO，旧版 rknn_server
// 驱动可能无法有效实现多实例负载均衡，导致只有 Core 0 满载。
static rknn_core_mask resolveCoreMaskFromEnv() {
  const char *env = std::getenv("RKNN_NPU_CORE");
  std::string s = env ? env : "";
  for (auto &c : s)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

  if (s == "0")
    return RKNN_NPU_CORE_0;
  if (s == "1")
    return RKNN_NPU_CORE_1;
  if (s == "2")
    return RKNN_NPU_CORE_2;
  if (s == "0_1" || s == "01")
    return RKNN_NPU_CORE_0_1;
  if (s == "0_1_2" || s == "012")
    return RKNN_NPU_CORE_0_1_2;
  if (s == "AUTO")
    return RKNN_NPU_CORE_AUTO;

  // 默认行为：显式轮询绑定，确保多个实例均匀分布在 3 个核心上
  static std::atomic<int> s_core_idx{0};
  int idx = s_core_idx.fetch_add(1) % 3;
  if (idx == 0)
    return RKNN_NPU_CORE_0;
  if (idx == 1)
    return RKNN_NPU_CORE_1;
  return RKNN_NPU_CORE_2;
}

static const char *coreMaskToString(rknn_core_mask m) {
  switch (m) {
  case RKNN_NPU_CORE_AUTO:
    return "AUTO (runtime decides)";
  case RKNN_NPU_CORE_0:
    return "CORE_0";
  case RKNN_NPU_CORE_1:
    return "CORE_1";
  case RKNN_NPU_CORE_2:
    return "CORE_2";
  case RKNN_NPU_CORE_0_1:
    return "CORE_0_1";
  case RKNN_NPU_CORE_0_1_2:
    return "CORE_0_1_2 (ALL 3 cores)";
  default:
    return "UNKNOWN";
  }
}

namespace fs = std::filesystem;

static std::string trimStr(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos)
    return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// 私有静态 helper: 从输出张量数组反推 YOLO 类别数.
//   单输出:   dim1/dim2 较大者为 boxes, 较小者 = 4 + nc (v5) 或 64 + nc (v8/v11
//   DFL). 多输出:   任一特征图的通道数 c = 4 + nc 或 64 + nc,
//   找到第一个合理的就返回. nc 不在 [1, 100] 视为不可信, 返回 0.
int RKNNInferEngine::DetectNumClassesFromOutputs(const rknn_tensor_attr *attrs,
                                                 uint32_t n_output) {
  if (!attrs || n_output == 0)
    return 0;
  if (n_output == 1) {
    uint32_t d1 = attrs[0].dims[1];
    uint32_t d2 = attrs[0].dims[2];
    uint32_t small = (d1 < d2) ? d1 : d2;
    if (small >= 64)
      return static_cast<int>(small - 64);
    else if (small >= 4)
      return static_cast<int>(small - 4);
    return 0;
  }
  for (uint32_t i = 0; i < n_output; ++i) {
    uint32_t c = attrs[i].dims[3];
    if (attrs[i].fmt == RKNN_TENSOR_NCHW) {
      c = attrs[i].dims[1];
    }
    int nc = 0;
    if (c >= 64)
      nc = static_cast<int>(c - 64);
    else if (c >= 4)
      nc = static_cast<int>(c - 4);
    if (nc > 0 && nc <= 100)
      return nc;
  }
  return 0;
}

// 公有静态 helper: 任何模块都可以读 sidecar 拿到类别名, 不需要构造 engine 实例.
//   优先级: .json > .labels > .txt > .names (都与 .rknn 同前缀)
//     - .json:          {"classes":[{"key":"..","label":".."}, ...]}
//     - .labels/.txt/.names:  每行一个英文名
//   num_classes <= 0 表示不校验行数, 原样返回.
//   找到任意一个并解析成功返回 true, 全部没找到返回 false.
//   out_en / out_cn 即便返回 false 也会被清空, 避免外部读到陈旧数据.
bool RKNNInferEngine::ReadSidecarClasses(const std::string &model_path,
                                         int num_classes,
                                         std::vector<std::string> &out_en,
                                         std::vector<std::string> &out_cn) {
  out_en.clear();
  out_cn.clear();

  fs::path p(model_path);
  fs::path base = p;
  base.replace_extension(""); // 去掉 .rknn

  std::vector<fs::path> candidates = {
      base.string() + ".json",
      base.string() + ".labels",
      base.string() + ".txt",
      base.string() + ".names",
  };

  for (const auto &cand : candidates) {
    std::error_code ec;
    if (!fs::exists(cand, ec) || ec)
      continue;

    std::cout << "[RKNNInferEngine] Loading class names from sidecar: " << cand
              << std::endl;

    std::vector<std::string> en, cn;

    std::string ext = cand.extension().string();
    if (ext == ".json") {
      std::ifstream f(cand);
      if (!f)
        continue;
      try {
        nlohmann::json j;
        f >> j;
        if (j.contains("classes") && j["classes"].is_array()) {
          for (const auto &c : j["classes"]) {
            std::string key = c.value("key", std::string());
            if (key.empty())
              continue;
            std::string label = c.value("label", std::string());
            en.push_back(key);
            cn.push_back(label.empty() ? key : label);
          }
        }
      } catch (const std::exception &e) {
        std::cerr << "[RKNNInferEngine] Failed to parse JSON sidecar: "
                  << e.what() << std::endl;
        continue;
      }
    } else {
      // .labels / .txt / .names: 纯文本, 每行一个英文类名
      std::ifstream f(cand);
      if (!f)
        continue;
      std::string line;
      while (std::getline(f, line)) {
        std::string t = trimStr(line);
        if (t.empty() || t[0] == '#')
          continue; // # 开头视为注释
        en.push_back(t);
        cn.push_back(t);
      }
    }

    if (en.empty()) {
      std::cerr << "[RKNNInferEngine] Sidecar " << cand
                << " parsed 0 classes, trying next" << std::endl;
      continue;
    }

    // sidecar 行数和模型 num_classes 不一致时, 警告并裁/补到模型维度
    if (num_classes > 0 && static_cast<int>(en.size()) != num_classes) {
      std::cerr << "[RKNNInferEngine] WARNING: sidecar has " << en.size()
                << " classes but model has " << num_classes
                << ", will adjust to match model" << std::endl;
      if (static_cast<int>(en.size()) > num_classes) {
        en.resize(num_classes);
        cn.resize(num_classes);
      } else {
        while (static_cast<int>(en.size()) < num_classes) {
          int i = static_cast<int>(en.size());
          en.push_back("class_" + std::to_string(i));
          cn.push_back("未知类别_" + std::to_string(i));
        }
      }
    }

    out_en = std::move(en);
    out_cn = std::move(cn);

    std::cout << "[RKNNInferEngine] Loaded " << out_en.size()
              << " class names from sidecar:";
    for (const auto &s : out_en)
      std::cout << " " << s;
    std::cout << std::endl;
    return true;
  }

  std::cout
      << "[RKNNInferEngine] No class names sidecar found near " << model_path
      << " (tried .json/.labels/.txt/.names), falling back to legacy profile"
      << std::endl;
  return false;
}

// 实例方法包装: LoadModel 里调用, 把结果存到成员变量里
bool RKNNInferEngine::LoadClassNamesFromSidecar(const std::string &model_path,
                                                int num_classes) {
  return ReadSidecarClasses(model_path, num_classes, class_names_en_,
                            class_names_cn_);
}

RKNNInferEngine::RKNNInferEngine()
    : ctx_(0), is_init_(false), input_attrs_(nullptr), output_attrs_(nullptr) {}

RKNNInferEngine::~RKNNInferEngine() {
  if (input_mem_) {
    rknn_destroy_mem(ctx_, input_mem_);
  }
  if (ctx_ > 0) {
    rknn_destroy(ctx_);
  }
  if (input_attrs_)
    delete[] input_attrs_;
  if (output_attrs_)
    delete[] output_attrs_;
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz) {
  unsigned char *data;
  int ret;
  data = NULL;
  if (NULL == fp) {
    return NULL;
  }
  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    std::cerr << "blob seek failure." << std::endl;
    return NULL;
  }
  data = (unsigned char *)malloc(sz);
  if (data == NULL) {
    std::cerr << "buffer malloc failure." << std::endl;
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char *load_model(const char *filename, int *model_size) {
  FILE *fp;
  unsigned char *data;
  fp = fopen(filename, "rb");
  if (NULL == fp) {
    std::cerr << "Open file " << filename << " failed." << std::endl;
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);
  data = load_data(fp, 0, size);
  fclose(fp);
  *model_size = size;
  return data;
}

bool RKNNInferEngine::LoadModel(const std::string &model_path) {
  std::cout << "[RKNNInferEngine] Loading model from: " << model_path
            << std::endl;

  int model_data_size = 0;
  unsigned char *model_data = load_model(model_path.c_str(), &model_data_size);
  if (model_data == nullptr) {
    std::cerr << "[RKNNInferEngine] ERROR: Model file not found or load failed!"
              << std::endl;
    return false;
  }

  // NPU 核心调度: 默认 RK_NPU_CORE_0_1_2 (RK3588 全部 3 核),
  // 用独立的 rknn_set_core_mask() API 在 init 后启用多核 (跨版本最稳),
  // 之后 rknn_run 会把任务调度到任意空闲核, 多路摄像头同时推理就能真正并行
  rknn_core_mask core_mask = resolveCoreMaskFromEnv();

  int ret = rknn_init(&ctx_, model_data, model_data_size, 0, NULL);
  free(model_data);

  if (ret < 0) {
    std::cerr << "[RKNNInferEngine] ERROR: rknn_init failed! ret = " << ret
              << std::endl;
    return false;
  }

  int mask_ret = rknn_set_core_mask(ctx_, core_mask);
  std::cout << "[RKNNInferEngine] NPU core mask = "
            << coreMaskToString(core_mask);
  if (const char *env = std::getenv("RKNN_NPU_CORE")) {
    std::cout << " (env RKNN_NPU_CORE='" << env << "')";
  }
  std::cout << ", rknn_set_core_mask ret=" << mask_ret << std::endl;
  if (mask_ret != 0) {
    std::cerr << "[RKNNInferEngine] WARNING: rknn_set_core_mask failed (ret="
              << mask_ret << "). 模型仍可工作, 但只会使用默认核." << std::endl;
  }

  // 获取输入输出个数
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret < 0) {
    std::cerr << "[RKNNInferEngine] ERROR: rknn_query in_out_num failed! ret = "
              << ret << std::endl;
    return false;
  }

  input_attrs_ = new rknn_tensor_attr[io_num_.n_input];
  for (uint32_t i = 0; i < io_num_.n_input; i++) {
    input_attrs_[i].index = i;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]),
               sizeof(rknn_tensor_attr));
  }

  output_attrs_ = new rknn_tensor_attr[io_num_.n_output];
  for (uint32_t i = 0; i < io_num_.n_output; i++) {
    output_attrs_[i].index = i;
    rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]),
               sizeof(rknn_tensor_attr));
    std::cout << "  Output[" << i << "]: name=" << output_attrs_[i].name
              << ", dims=[";
    for (uint32_t d = 0; d < output_attrs_[i].n_dims; d++) {
      std::cout << output_attrs_[i].dims[d]
                << (d + 1 < output_attrs_[i].n_dims ? ", " : "");
    }
    std::cout << "], fmt=" << output_attrs_[i].fmt
              << ", type=" << output_attrs_[i].type << std::endl;
  }

  if (input_attrs_[0].fmt == RKNN_TENSOR_NCHW) {
    model_channel_ = input_attrs_[0].dims[1];
    model_height_ = input_attrs_[0].dims[2];
    model_width_ = input_attrs_[0].dims[3];
  } else { // NHWC
    model_height_ = input_attrs_[0].dims[1];
    model_width_ = input_attrs_[0].dims[2];
    model_channel_ = input_attrs_[0].dims[3];
  }

  std::cout << "[RKNNInferEngine] Model initialized successfully." << std::endl;
  std::cout << "  Input shape: " << model_width_ << "x" << model_height_ << "x"
            << model_channel_ << std::endl;
  std::cout << "  Model Type: Quantized (INT8/UINT8)" << std::endl;

  // --- RGA ZERO COPY ALLOCATION ---
  // 分配可与 NPU / RGA 共享的连续物理内存
  input_mem_ =
      rknn_create_mem(ctx_, model_width_ * model_height_ * model_channel_);
  if (!input_mem_) {
    std::cerr
        << "[RKNNInferEngine] ERROR: rknn_create_mem for zero-copy failed!"
        << std::endl;
    return false;
  }
  int set_ret = rknn_set_io_mem(ctx_, input_mem_, &input_attrs_[0]);
  if (set_ret < 0) {
    std::cerr << "[RKNNInferEngine] ERROR: rknn_set_io_mem failed! ret="
              << set_ret << std::endl;
    return false;
  }
  std::cout
      << "[RKNNInferEngine] Zero-copy input memory allocated and bound to NPU."
      << std::endl;
  // --------------------------------

  // 提前从输出张量形状反推 num_classes，让 AlertService
  // 不用等到第一次推理就能拿到 单输出：dim1 / dim2 较大者为 boxes，较小者 = 4 +
  // nc 或 64 + nc 多输出：任一特征图的通道数 c = 4 + nc 或 64 + nc
  if (io_num_.n_output == 1) {
    uint32_t d1 = output_attrs_[0].dims[1];
    uint32_t d2 = output_attrs_[0].dims[2];
    uint32_t big = (d1 > d2) ? d1 : d2;
    uint32_t small = (d1 < d2) ? d1 : d2;
    if (small >= 64)
      num_classes_ = static_cast<int>(small - 64);
    else if (small >= 4)
      num_classes_ = static_cast<int>(small - 4);
    std::cout << "  Detected num_classes=" << num_classes_
              << " from output[0] dims=[" << d1 << "," << d2 << "]"
              << std::endl;
  } else {
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
      uint32_t c = output_attrs_[i].dims[3];
      if (output_attrs_[i].fmt == RKNN_TENSOR_NCHW) {
        c = output_attrs_[i].dims[1];
      }
      int nc = 0;
      if (c >= 64)
        nc = static_cast<int>(c - 64);
      else if (c >= 4)
        nc = static_cast<int>(c - 4);
      if (nc > 0 && nc <= 100) {
        num_classes_ = nc;
        std::cout << "  Detected num_classes=" << num_classes_
                  << " from output[" << i << "] (multi-output)" << std::endl;
        break;
      }
    }
  }

  // 模型加载完毕后, 尝试读取同目录下的类别名 sidecar.
  // 找到 -> 直接覆盖内置 profile (任意类别数都行);
  // 找不到 -> num_classes_ 仍可被 AlertService 用来选取 3 / 4 类兼容 profile.
  LoadClassNamesFromSidecar(model_path, num_classes_);

  is_init_ = true;
  return true;
}

float RKNNInferEngine::CalculateIoU(const cv::Rect &a, const cv::Rect &b) {
  int inter_area = (a & b).area();
  if (inter_area == 0)
    return 0.0f;
  int union_area = a.area() + b.area() - inter_area;
  return static_cast<float>(inter_area) / union_area;
}

void RKNNInferEngine::NMS(std::vector<DetectResult> &boxes, float nms_thresh) {
  std::sort(boxes.begin(), boxes.end(),
            [](const DetectResult &a, const DetectResult &b) {
              return a.confidence > b.confidence;
            });

  std::vector<bool> is_suppressed(boxes.size(), false);
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (is_suppressed[i])
      continue;
    for (size_t j = i + 1; j < boxes.size(); ++j) {
      if (is_suppressed[j])
        continue;
      if (boxes[i].class_id == boxes[j].class_id) {
        if (CalculateIoU(boxes[i].box, boxes[j].box) > nms_thresh) {
          is_suppressed[j] = true;
        }
      }
    }
  }

  std::vector<DetectResult> result;
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (!is_suppressed[i]) {
      result.push_back(boxes[i]);
    }
  }
  boxes = result;
}

inline float fast_sigmoid(float x) {
  if (x < -10.0f)
    return 0.0f;
  if (x > 10.0f)
    return 1.0f;
  return 1.0f / (1.0f + std::exp(-x));
}

#if defined(__ARM_NEON) || defined(__aarch64__)
// NEON 快速指数近似，专为 DFL (x <= 0) 场景优化
// e^x = 1 / (1 - x + x^2/2 - x^3/6 + x^4/24 - x^5/120) for x <= 0
// 此多项式在 [-5, 0] 范围内精度极高，超出 -5 的尾部虽然偏大但总体已趋近于
// 0，不影响 softmax
inline float32x4_t fast_exp_neon(float32x4_t x) {
  // 限制下限以避免溢出
  x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
  x = vminq_f32(x, vdupq_n_f32(0.0f));

  float32x4_t y = vnegq_f32(x); // y = -x, 故 y >= 0
  float32x4_t y2 = vmulq_f32(y, y);
  float32x4_t y3 = vmulq_f32(y2, y);
  float32x4_t y4 = vmulq_f32(y2, y2);
  float32x4_t y5 = vmulq_f32(y4, y);

  float32x4_t poly = vdupq_n_f32(1.0f);
  poly = vaddq_f32(poly, y);
  poly = vmlaq_f32(poly, y2, vdupq_n_f32(1.0f / 2.0f));
  poly = vmlaq_f32(poly, y3, vdupq_n_f32(1.0f / 6.0f));
  poly = vmlaq_f32(poly, y4, vdupq_n_f32(1.0f / 24.0f));
  poly = vmlaq_f32(poly, y5, vdupq_n_f32(1.0f / 120.0f));

  // e^x = 1.0f / poly
  float32x4_t recip = vrecpeq_f32(poly);
  recip = vmulq_f32(recip, vrecpsq_f32(poly, recip)); // 1st Newton-Raphson step
  recip = vmulq_f32(
      recip, vrecpsq_f32(poly, recip)); // 2nd Newton-Raphson step for precision
  return recip;
}

static float compute_dfl_dist(const float *dfl_ptr) {
  float32x4_t v_dfl0 = vld1q_f32(dfl_ptr);
  float32x4_t v_dfl1 = vld1q_f32(dfl_ptr + 4);
  float32x4_t v_dfl2 = vld1q_f32(dfl_ptr + 8);
  float32x4_t v_dfl3 = vld1q_f32(dfl_ptr + 12);

  float32x4_t v_max = vmaxq_f32(v_dfl0, v_dfl1);
  v_max = vmaxq_f32(v_max, v_dfl2);
  v_max = vmaxq_f32(v_max, v_dfl3);

#if defined(__aarch64__)
  float max_val = vmaxvq_f32(v_max);
#else
  float32x2_t v_max_half = vmax_f32(vget_low_f32(v_max), vget_high_f32(v_max));
  v_max_half = vpmax_f32(v_max_half, v_max_half);
  float max_val = vget_lane_f32(v_max_half, 0);
#endif

  float32x4_t v_max_vec = vdupq_n_f32(max_val);
  float32x4_t v_exp0 = fast_exp_neon(vsubq_f32(v_dfl0, v_max_vec));
  float32x4_t v_exp1 = fast_exp_neon(vsubq_f32(v_dfl1, v_max_vec));
  float32x4_t v_exp2 = fast_exp_neon(vsubq_f32(v_dfl2, v_max_vec));
  float32x4_t v_exp3 = fast_exp_neon(vsubq_f32(v_dfl3, v_max_vec));

  float32x4_t v_sum_exp =
      vaddq_f32(vaddq_f32(v_exp0, v_exp1), vaddq_f32(v_exp2, v_exp3));
#if defined(__aarch64__)
  float sum_exp = vaddvq_f32(v_sum_exp);
#else
  float32x2_t v_sum_exp_half =
      vadd_f32(vget_low_f32(v_sum_exp), vget_high_f32(v_sum_exp));
  v_sum_exp_half = vpadd_f32(v_sum_exp_half, v_sum_exp_half);
  float sum_exp = vget_lane_f32(v_sum_exp_half, 0);
#endif

  float32x4_t v_sum = vdupq_n_f32(0.0f);
  const float32x4_t v_idx0 = {0.0f, 1.0f, 2.0f, 3.0f};
  const float32x4_t v_idx1 = {4.0f, 5.0f, 6.0f, 7.0f};
  const float32x4_t v_idx2 = {8.0f, 9.0f, 10.0f, 11.0f};
  const float32x4_t v_idx3 = {12.0f, 13.0f, 14.0f, 15.0f};

  v_sum = vmlaq_f32(v_sum, v_exp0, v_idx0);
  v_sum = vmlaq_f32(v_sum, v_exp1, v_idx1);
  v_sum = vmlaq_f32(v_sum, v_exp2, v_idx2);
  v_sum = vmlaq_f32(v_sum, v_exp3, v_idx3);

#if defined(__aarch64__)
  float dist = vaddvq_f32(v_sum);
#else
  float32x2_t v_dist_half = vadd_f32(vget_low_f32(v_sum), vget_high_f32(v_sum));
  v_dist_half = vpadd_f32(v_dist_half, v_dist_half);
  float dist = vget_lane_f32(v_dist_half, 0);
#endif

  return dist / sum_exp;
}
#else
static float compute_dfl_dist(const float *dfl_ptr) {
  float max_val = dfl_ptr[0];
  for (int i = 1; i < 16; ++i) {
    if (dfl_ptr[i] > max_val)
      max_val = dfl_ptr[i];
  }
  float sum = 0.0f;
  float exp_buf[16];
  for (int i = 0; i < 16; ++i) {
    exp_buf[i] = std::exp(dfl_ptr[i] - max_val);
    sum += exp_buf[i];
  }
  float dist = 0.0f;
  for (int i = 0; i < 16; ++i) {
    dist += (exp_buf[i] / sum) * static_cast<float>(i);
  }
  return dist;
}
#endif

static inline float get_tensor_float(const rknn_output &output,
                                     const rknn_tensor_attr &attr,
                                     size_t index) {
  if (output.want_float) {
    return static_cast<const float *>(output.buf)[index];
  }
  if (attr.type == RKNN_TENSOR_INT8) {
    int8_t val = static_cast<const int8_t *>(output.buf)[index];
    return (static_cast<float>(val) - attr.zp) * attr.scale;
  } else if (attr.type == RKNN_TENSOR_UINT8) {
    uint8_t val = static_cast<const uint8_t *>(output.buf)[index];
    return (static_cast<float>(val) - attr.zp) * attr.scale;
  } else if (attr.type == RKNN_TENSOR_INT32) {
    int32_t val = static_cast<const int32_t *>(output.buf)[index];
    return (static_cast<float>(val) - attr.zp) * attr.scale;
  } else {
    return static_cast<const float *>(output.buf)[index];
  }
}

void RKNNInferEngine::PostProcessYOLOv11(rknn_output *outputs,
                                         std::vector<DetectResult> &results) {
  uint32_t dim1 = output_attrs_[0].dims[1];
  uint32_t dim2 = output_attrs_[0].dims[2];

  int num_boxes = (dim1 > dim2) ? dim1 : dim2;
  int num_attrs = (dim1 < dim2) ? dim1 : dim2;

  bool is_dfl = (num_attrs >= 64);
  int num_classes = is_dfl ? (num_attrs - 64) : (num_attrs - 4);
  int class_offset = is_dfl ? 64 : 4;

  // 记录当前模型的类别数，供 AlertService 选取合适的类别名 / 配色 profile
  if (num_classes > 0 && num_classes <= 100) {
    num_classes_ = num_classes;
  }

  bool is_transposed = (dim2 == num_boxes);

  // ---- 诊断 + 自适应激活判断 ----
  // RKNN 转换 YOLO 时 sigmoid 可能已被融合进模型，此时输出的类别分数已经是
  // [0,1] 概率。 若后处理再套一层 sigmoid，背景处的 0 会被变成 0.5，越过 0.45
  // 阈值，导致满屏 50% 误检。 这里通过采样判断输出究竟是 raw logits
  // 还是已归一化概率。
  if (!cls_activation_checked_) {
    cls_activation_checked_ = true;
    float cmin = 1e9f, cmax = -1e9f;
    for (int c = 0; c < num_classes; ++c) {
      for (int i = 0; i < num_boxes; i += 137) {
        size_t idx = is_transposed ? ((class_offset + c) * num_boxes + i)
                                   : (i * num_attrs + class_offset + c);
        float v = get_tensor_float(outputs[0], output_attrs_[0], idx);
        if (v < cmin)
          cmin = v;
        if (v > cmax)
          cmax = v;
      }
    }
    // logits 必然存在明显负值；若全部落在 [0,1] 附近，说明模型内部已含 sigmoid
    model_has_sigmoid_ = (cmin <= cmax) && (cmin >= -0.05f && cmax <= 1.05f);

    std::cout << "[RKNNInferEngine] PostProcess(single) dims=" << dim1 << "x"
              << dim2 << ", boxes=" << num_boxes << ", attrs=" << num_attrs
              << ", classes=" << num_classes << ", dfl=" << is_dfl
              << ", transposed=" << is_transposed << std::endl;
    std::cout << "[RKNNInferEngine] Class-score raw range = [" << cmin << ", "
              << cmax << "] -> "
              << (model_has_sigmoid_
                      ? "model ALREADY has sigmoid, skip extra sigmoid"
                      : "model outputs raw logits, apply sigmoid")
              << std::endl;
  }
  // --------------------------------

  const float conf_threshold = 0.45f;
  std::vector<DetectResult> candidates;

  for (int i = 0; i < num_boxes; ++i) {
    float max_class_score = 0.0f;
    int class_id = -1;

    for (int c = 0; c < num_classes; ++c) {
      size_t idx = is_transposed ? ((class_offset + c) * num_boxes + i)
                                 : (i * num_attrs + class_offset + c);
      float raw_score = get_tensor_float(outputs[0], output_attrs_[0], idx);
      float score = model_has_sigmoid_ ? raw_score : fast_sigmoid(raw_score);
      if (score > max_class_score) {
        max_class_score = score;
        class_id = c;
      }
    }

    if (max_class_score > conf_threshold) {
      float cx, cy, w, h;
      if (is_dfl) {
        float dfl_logits[64];
        for (int ch = 0; ch < 64; ++ch) {
          size_t idx =
              is_transposed ? (ch * num_boxes + i) : (i * num_attrs + ch);
          dfl_logits[ch] = get_tensor_float(outputs[0], output_attrs_[0], idx);
        }
        float d_left = compute_dfl_dist(dfl_logits + 0);
        float d_top = compute_dfl_dist(dfl_logits + 16);
        float d_right = compute_dfl_dist(dfl_logits + 32);
        float d_bottom = compute_dfl_dist(dfl_logits + 48);

        cx = (d_right - d_left) / 2.0f;
        cy = (d_bottom - d_top) / 2.0f;
        w = d_right + d_left;
        h = d_bottom + d_top;
      } else {
        size_t idx0 = is_transposed ? (0 * num_boxes + i) : (i * num_attrs + 0);
        size_t idx1 = is_transposed ? (1 * num_boxes + i) : (i * num_attrs + 1);
        size_t idx2 = is_transposed ? (2 * num_boxes + i) : (i * num_attrs + 2);
        size_t idx3 = is_transposed ? (3 * num_boxes + i) : (i * num_attrs + 3);
        cx = get_tensor_float(outputs[0], output_attrs_[0], idx0);
        cy = get_tensor_float(outputs[0], output_attrs_[0], idx1);
        w = get_tensor_float(outputs[0], output_attrs_[0], idx2);
        h = get_tensor_float(outputs[0], output_attrs_[0], idx3);
      }

      if (w <= 1.0f && h <= 1.0f) {
        cx *= model_width_;
        cy *= model_height_;
        w *= model_width_;
        h *= model_height_;
      }

      float x = cx - w / 2.0f;
      float y = cy - h / 2.0f;

      DetectResult res;
      res.class_id = class_id;
      res.confidence = max_class_score;
      res.box.x = static_cast<int>(x);
      res.box.y = static_cast<int>(y);
      res.box.width = static_cast<int>(w);
      res.box.height = static_cast<int>(h);

      candidates.push_back(res);
    }
  }

  NMS(candidates, 0.45f);
  results = candidates;

  if (!results.empty()) {
    // Log removed to avoid console spam for static objects.
    // Alarms and box drawing are handled by AlertService for dynamic objects.
  }
}

void RKNNInferEngine::PostProcessMultiOutput(
    rknn_output *outputs, std::vector<DetectResult> &results) {
  const float conf_threshold = 0.45f;
  std::vector<DetectResult> candidates;

  // ---- 诊断 + 自适应激活判断 (同单输出逻辑) ----
  if (!cls_activation_checked_) {
    cls_activation_checked_ = true;
    float cmin = 1e9f, cmax = -1e9f;
    for (uint32_t k = 0; k < io_num_.n_output; ++k) {
      uint32_t hh = output_attrs_[k].dims[1];
      uint32_t ww = output_attrs_[k].dims[2];
      uint32_t cc = output_attrs_[k].dims[3];
      bool nchw = (output_attrs_[k].fmt == RKNN_TENSOR_NCHW);
      if (nchw) {
        cc = output_attrs_[k].dims[1];
        hh = output_attrs_[k].dims[2];
        ww = output_attrs_[k].dims[3];
      }
      bool dfl = (cc >= 64);
      int nc = dfl ? static_cast<int>(cc) - 64 : static_cast<int>(cc) - 4;
      int off = dfl ? 64 : 4;
      if (nc <= 0 || nc > 100)
        continue;

      size_t plane = static_cast<size_t>(hh) * ww;
      for (int cl = 0; cl < nc; ++cl) {
        for (size_t p = 0; p < plane; p += 131) {
          size_t idx = nchw ? (static_cast<size_t>(off + cl) * plane + p)
                            : (p * cc + off + cl);
          float v = get_tensor_float(outputs[k], output_attrs_[k], idx);
          if (v < cmin)
            cmin = v;
          if (v > cmax)
            cmax = v;
        }
      }
    }
    model_has_sigmoid_ = (cmin <= cmax) && (cmin >= -0.05f && cmax <= 1.05f);
    std::cout << "[RKNNInferEngine] Multi-output class raw range = [" << cmin
              << ", " << cmax << "] -> "
              << (model_has_sigmoid_
                      ? "model ALREADY has sigmoid, skip extra sigmoid"
                      : "model outputs raw logits, apply sigmoid")
              << std::endl;
  }
  // ---------------------------------------------

  for (uint32_t k = 0; k < io_num_.n_output; ++k) {
    uint32_t h = output_attrs_[k].dims[1];
    uint32_t w = output_attrs_[k].dims[2];
    uint32_t c = output_attrs_[k].dims[3];

    if (output_attrs_[k].fmt == RKNN_TENSOR_NCHW) {
      c = output_attrs_[k].dims[1];
      h = output_attrs_[k].dims[2];
      w = output_attrs_[k].dims[3];
    }

    int grid_h = h;
    int grid_w = w;
    if (grid_h <= 0 || grid_w <= 0)
      continue;
    int stride = model_height_ / grid_h;

    bool is_dfl = (c >= 64);
    int num_classes = is_dfl ? (c - 64) : (c - 4);
    int class_offset = is_dfl ? 64 : 4;

    if (num_classes <= 0 || num_classes > 100)
      continue;

    // 把首个合法特征图的类别数缓存起来，给 AlertService / WebServer 使用
    if (num_classes_ == 0) {
      num_classes_ = num_classes;
    }

    for (int gh = 0; gh < grid_h; ++gh) {
      for (int gw = 0; gw < grid_w; ++gw) {
        int offset = (gh * grid_w + gw) * c;

        float max_score = 0.0f;
        int class_id = -1;
        for (int cl = 0; cl < num_classes; ++cl) {
          size_t idx =
              (output_attrs_[k].fmt == RKNN_TENSOR_NCHW)
                  ? ((class_offset + cl) * grid_h * grid_w + gh * grid_w + gw)
                  : (offset + class_offset + cl);

          float raw_score = get_tensor_float(outputs[k], output_attrs_[k], idx);
          float score =
              model_has_sigmoid_ ? raw_score : fast_sigmoid(raw_score);
          if (score > max_score) {
            max_score = score;
            class_id = cl;
          }
        }

        if (max_score > conf_threshold) {
          float cx = 0.0f, cy = 0.0f, bw = 0.0f, bh = 0.0f;

          if (is_dfl) {
            float dfl_logits[64];
            if (output_attrs_[k].fmt == RKNN_TENSOR_NCHW) {
              for (int ch = 0; ch < 64; ++ch) {
                size_t idx = ch * grid_h * grid_w + gh * grid_w + gw;
                dfl_logits[ch] =
                    get_tensor_float(outputs[k], output_attrs_[k], idx);
              }
            } else {
              for (int ch = 0; ch < 64; ++ch) {
                size_t idx = offset + ch;
                dfl_logits[ch] =
                    get_tensor_float(outputs[k], output_attrs_[k], idx);
              }
            }

            float d_left = compute_dfl_dist(dfl_logits + 0);
            float d_top = compute_dfl_dist(dfl_logits + 16);
            float d_right = compute_dfl_dist(dfl_logits + 32);
            float d_bottom = compute_dfl_dist(dfl_logits + 48);

            float x1 = (gw + 0.5f - d_left) * stride;
            float y1 = (gh + 0.5f - d_top) * stride;
            float x2 = (gw + 0.5f + d_right) * stride;
            float y2 = (gh + 0.5f + d_bottom) * stride;

            cx = (x1 + x2) / 2.0f;
            cy = (y1 + y2) / 2.0f;
            bw = x2 - x1;
            bh = y2 - y1;
          } else {
            size_t idx0 = (output_attrs_[k].fmt == RKNN_TENSOR_NCHW)
                              ? (0 * grid_h * grid_w + gh * grid_w + gw)
                              : (offset + 0);
            size_t idx1 = (output_attrs_[k].fmt == RKNN_TENSOR_NCHW)
                              ? (1 * grid_h * grid_w + gh * grid_w + gw)
                              : (offset + 1);
            size_t idx2 = (output_attrs_[k].fmt == RKNN_TENSOR_NCHW)
                              ? (2 * grid_h * grid_w + gh * grid_w + gw)
                              : (offset + 2);
            size_t idx3 = (output_attrs_[k].fmt == RKNN_TENSOR_NCHW)
                              ? (3 * grid_h * grid_w + gh * grid_w + gw)
                              : (offset + 3);

            float dx = get_tensor_float(outputs[k], output_attrs_[k], idx0);
            float dy = get_tensor_float(outputs[k], output_attrs_[k], idx1);
            float dw = get_tensor_float(outputs[k], output_attrs_[k], idx2);
            float dh = get_tensor_float(outputs[k], output_attrs_[k], idx3);

            cx = (gw + fast_sigmoid(dx)) * stride;
            cy = (gh + fast_sigmoid(dy)) * stride;
            bw = std::exp(dw) * stride;
            bh = std::exp(dh) * stride;

            if (bw <= 0 || bh <= 0 || bw > model_width_ * 2 ||
                bh > model_height_ * 2) {
              cx = dx;
              cy = dy;
              bw = dw;
              bh = dh;
              if (bw <= 1.0f && bh <= 1.0f) {
                cx *= model_width_;
                cy *= model_height_;
                bw *= model_width_;
                bh *= model_height_;
              }
            }
          }

          DetectResult res;
          res.class_id = class_id;
          res.confidence = max_score;
          res.box.x = static_cast<int>(cx - bw / 2.0f);
          res.box.y = static_cast<int>(cy - bh / 2.0f);
          res.box.width = static_cast<int>(bw);
          res.box.height = static_cast<int>(bh);
          candidates.push_back(res);
        }
      }
    }
  }

  NMS(candidates, 0.45f);
  results = candidates;
}

std::vector<DetectResult> RKNNInferEngine::Inference(const cv::Mat &frame) {
  std::vector<DetectResult> results;
  if (!is_init_ || frame.empty())
    return results;

  // [关键修复] 防止多个 Worker 线程同时 pop 到同一个摄像头的任务，
  // 并发调用同一个 engine 的 Inference，导致 rknn_run 崩溃。
  std::lock_guard<std::mutex> infer_lock(infer_mutex_);

  ScopedTimer t_total("Infer.Total");

  // 1. 预处理 (使用 RGA 硬件加速 + 零拷贝)
  // cv::Mat (BGR) -> RGA -> input_mem_ (RGB, Resized)
  if (!input_mem_) {
    std::cerr << "[RKNNInferEngine] Zero-copy input memory is missing!" << std::endl;
    return results;
  }

  {
    ScopedTimer t_pre("Infer.RGA_Preprocess");
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)frame.data, frame.cols, frame.rows, RK_FORMAT_BGR_888);
    rga_buffer_t dst = wrapbuffer_fd(input_mem_->fd, model_width_, model_height_, RK_FORMAT_RGB_888);

    IM_STATUS STATUS = imresize(src, dst);
    if (STATUS != IM_STATUS_SUCCESS) {
      std::cerr << "[RKNNInferEngine] RGA imresize failed: " << imStrError(STATUS) << std::endl;
      return results;
    }
  }

  // 3. 执行推理 (NPU 实际干活，单次最大耗时点)
  {
    ScopedTimer t_run("Infer.rknn_run");
    int ret = rknn_run(ctx_, NULL);
    if (ret < 0) {
      std::cerr << "[RKNNInferEngine] rknn_run failed! ret=" << ret
                << std::endl;
      return results;
    }
  }

  // 4. 动态获取模型的所有输出张量 (兼容单输出与多特征图输出)
  std::vector<rknn_output> outputs(io_num_.n_output);
  memset(outputs.data(), 0, sizeof(rknn_output) * io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_output; i++) {
    outputs[i].want_float = 1; // 转换为 float
  }
  {
    ScopedTimer t_get("Infer.outputs_get");
    int ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), NULL);
    if (ret < 0) {
      std::cerr << "[RKNNInferEngine] rknn_outputs_get failed! ret=" << ret
                << std::endl;
      return results;
    }
  }

  // 5. 后处理解码 (区分合并后的单输出张量与未合并的多输出特征图)
  {
    ScopedTimer t_post("Infer.Postprocess");
    if (io_num_.n_output == 1) {
      PostProcessYOLOv11(outputs.data(), results);
    } else {
      PostProcessMultiOutput(outputs.data(), results);
    }
  }

  rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());

  // 将检测框缩贴回原始图像尺寸
  float scale_x = static_cast<float>(frame.cols) / model_width_;
  float scale_y = static_cast<float>(frame.rows) / model_height_;

  for (auto &res : results) {
    res.box.x = static_cast<int>(res.box.x * scale_x);
    res.box.y = static_cast<int>(res.box.y * scale_y);
    res.box.width = static_cast<int>(res.box.width * scale_x);
    res.box.height = static_cast<int>(res.box.height * scale_y);

    // 边界保护
    res.box.x = std::max(0, res.box.x);
    res.box.y = std::max(0, res.box.y);
    if (res.box.x + res.box.width > frame.cols)
      res.box.width = frame.cols - res.box.x;
    if (res.box.y + res.box.height > frame.rows)
      res.box.height = frame.rows - res.box.y;
  }

  return results;
}
