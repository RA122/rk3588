# 云帆 AI 算法盒子 (AILogicBox)

本项目是基于瑞芯微 RK3588 打造的边缘 AI 算法盒子框架。
用于多路视频流的并发接入、NPU 加速推理（支持识别落石、地面凹陷、滑坡等地质灾害）并上报云平台。

## 系统架构
1. **多媒体加速**：利用 RK MPP (Media Process Platform) 进行多路 RTSP/H.264/H.265 硬件解码。利用 RGA (Raster Graphic Acceleration) 进行图像缩放和颜色空间转换。
2. **AI 推理**：利用 RKNN-Toolkit2 将 YOLO/分割模型运行在自带的 6TOPS NPU 上。
3. **报警逻辑**：通过 Paho-MQTT 上报 `ai_box/alerts` 主题，并将异常帧图像与短视频持久化至本地存储。

## 目录结构
- `config/` - 运行配置文件，包括 RTSP 源地址，MQTT 平台信息，任务类型与防抖帧率等。
- `include/` - C++ 核心接口头文件 (`VideoManager`, `RKNNInferEngine`, `AlertService`)。
- `src/` - 具体代码实现模块与系统入口 `main.cpp`。
- `models/` - 存放转化好的 `.rknn` 模型文件。

## 编译依赖说明 (板端环境)
在RK3588 Debian/Ubuntu 板卡上，需安装以下依赖：
```bash
sudo apt update
sudo apt install cmake gcc g++ build-essential libopencv-dev
sudo apt install libpaho-mqtt-dev
# 注意：Rockchip的 mpp, rga 和 rknn-toolkit2 运行时库通常在官方系统镜像中已预置，或需要从瑞芯微官方/野火资料中获取对应 deb 包安装。
```

## 编译方法
```bash
mkdir build && cd build
cmake ..
make -j4
```

## 运行
```bash
./ai_box
```
