import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.110.91', username='cat', password='temppwd', timeout=5)

cpp_code = """
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    setenv("GST_DEBUG", "rtspsrc:5", 1);
    std::string pipeline = "rtspsrc location=rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/102 latency=100 protocols=tcp ! decodebin ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! video/x-raw,format=BGR ! appsink drop=false max-buffers=1";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (cap.isOpened()) {
        cap.grab();
        cv::Mat bgr;
        if (cap.retrieve(bgr)) {
            std::cout << "BGR shape: " << bgr.rows << "x" << bgr.cols << " channels: " << bgr.channels() << std::endl;
        } else {
            std::cout << "Failed to retrieve BGR." << std::endl;
        }
    } else {
        std::cout << "Failed to open pipeline." << std::endl;
    }
    return 0;
}
"""
sftp = ssh.open_sftp()
with sftp.file('test_bgr.cpp', 'w') as f:
    f.write(cpp_code)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("g++ test_bgr.cpp -o test_bgr `pkg-config --cflags --libs opencv4` && ./test_bgr")
print("STDOUT:")
print(stdout.read().decode())
print("STDERR:")
print(stderr.read().decode())
ssh.close()
