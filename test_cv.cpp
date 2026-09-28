#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>

int main() {
    _putenv("OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|buffer_size;102400");
    cv::VideoCapture cap("https://stream.bova.tech:8443/live/dd15fe80ba0e11f0f2e0_1_sub/hls.m3u8?expire=1796008345&id=4918359928433&lt=lt.aa97c77a50034cde74117eb872e6d19ec301bb1c1abd7e898c13d74a102ea25f", cv::CAP_FFMPEG);
    std::cout << "With options: " << cap.isOpened() << std::endl;
    
    _putenv("OPENCV_FFMPEG_CAPTURE_OPTIONS=");
    cv::VideoCapture cap2("https://stream.bova.tech:8443/live/dd15fe80ba0e11f0f2e0_1_sub/hls.m3u8?expire=1796008345&id=4918359928433&lt=lt.aa97c77a50034cde74117eb872e6d19ec301bb1c1abd7e898c13d74a102ea25f", cv::CAP_FFMPEG);
    std::cout << "Without options: " << cap2.isOpened() << std::endl;
    return 0;
}
