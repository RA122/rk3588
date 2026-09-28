import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.110.91', username='cat', password='temppwd', timeout=5)

cpp_code = """
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string pipeline = "rtspsrc location=rtsp://admin:Yunfan%402025@192.168.110.86:554/Streaming/Channels/102 latency=0 ! decodebin ! videoconvert ! video/x-raw,format=NV12 ! appsink drop=true max-buffers=1";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (cap.isOpened()) {
        cap.grab();
        cv::Mat f, bgr;
        if (cap.retrieve(f)) {
            std::cout << "Shape: " << f.rows << "x" << f.cols << " channels: " << f.channels() << std::endl;
            if (f.channels() == 1) {
                cv::cvtColor(f, bgr, cv::COLOR_YUV2BGR_NV12);
                std::cout << "BGR shape: " << bgr.rows << "x" << bgr.cols << " channels: " << bgr.channels() << std::endl;
                cv::imwrite("test.jpg", bgr);
                std::cout << "Saved test.jpg. Empty? " << bgr.empty() << std::endl;
            }
        }
    }
    return 0;
}
"""

sftp = ssh.open_sftp()
with sftp.file('test_cv.cpp', 'w') as f:
    f.write(cpp_code)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("g++ test_cv.cpp -o test_cv `pkg-config --cflags --libs opencv4` && ./test_cv")
print("STDOUT:")
print(stdout.read().decode())
print("STDERR:")
print(stderr.read().decode())
ssh.close()
