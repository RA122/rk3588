cmake_minimum_required(VERSION 3.10)
project(ai_box CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
# 在真实的板端环境中，需要链接 OpenCV, Rockchip MPP, RGA, RKNN API 和 Paho-MQTT
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/rknn # (在鲁班猫板卡上的实际路径)
    /usr/include/rockchip # (MPP/RGA头文件路径)
    /usr/include/rockchip/rga
    /usr/include/rga
)

# Source files
set(SOURCES
    src/main.cpp
    src/VideoManager.cpp
    src/RKNNInferEngine.cpp
    src/AlertService.cpp
    src/WebServer.cpp
)

add_executable(ai_box ${SOURCES})

# 链接库文件
target_link_libraries(ai_box
    ${OpenCV_LIBS}
    pthread
    rknn_api
    # rockchip_mpp
    rga
    paho-mqtt3c  # 异步 MQTT 客户端库
)
