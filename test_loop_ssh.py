import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.110.91', username='cat', password='temppwd', timeout=5)

cpp_code = """
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::string pipeline = "rtspsrc location=rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/102 latency=0 ! decodebin ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! video/x-raw,format=BGR ! appsink drop=false max-buffers=1";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (cap.isOpened()) {
        for(int i = 0; i < 5; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (cap.grab()) {
                cv::Mat bgr;
                if (cap.retrieve(bgr)) {
                    std::cout << "Grabbed frame " << i << " shape: " << bgr.rows << "x" << bgr.cols << std::endl;
                } else {
                    std::cout << "Retrieve failed on frame " << i << std::endl;
                }
            } else {
                std::cout << "Grab failed on frame " << i << std::endl;
            }
        }
    } else {
        std::cout << "Failed to open pipeline." << std::endl;
    }
    return 0;
}
"""
sftp = ssh.open_sftp()
with sftp.file('test_loop.cpp', 'w') as f:
    f.write(cpp_code)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("g++ test_loop.cpp -o test_loop `pkg-config --cflags --libs opencv4` && ./test_loop")
print("STDOUT:")
print(stdout.read().decode())
print("STDERR:")
print(stderr.read().decode())
ssh.close()
