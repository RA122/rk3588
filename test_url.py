import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.110.91', username='cat', password='temppwd', timeout=5)

cpp_code = """
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string url = "https://stream.bova.tech:8443/live/dd15fe80ba0e11f0f2e0_1_sub/hls.m3u8?expire=1796008345&id=4918359928433&lt=lt.aa97c77a50034cde74117eb872e6d19ec301bb1c1abd7e898c13d74a102ea25f";
    std::string gstreamer = "uridecodebin uri=\\"" + url + "\\" ! videoconvert ! video/x-raw,format=BGR ! appsink";
    std::cout << "Testing uridecodebin..." << std::endl;
    cv::VideoCapture cap1(gstreamer, cv::CAP_GSTREAMER);
    std::cout << "GStreamer opened: " << cap1.isOpened() << std::endl;

    std::cout << "Testing FFmpeg fallback..." << std::endl;
    cv::VideoCapture cap2(url, cv::CAP_FFMPEG);
    std::cout << "FFmpeg opened: " << cap2.isOpened() << std::endl;

    return 0;
}
"""

sftp = ssh.open_sftp()
with sftp.file('test_url_cv.cpp', 'w') as f:
    f.write(cpp_code)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("g++ test_url_cv.cpp -o test_url_cv `pkg-config --cflags --libs opencv4` && ./test_url_cv")
print("STDOUT:")
print(stdout.read().decode())
print("STDERR:")
print(stderr.read().decode())
ssh.close()
