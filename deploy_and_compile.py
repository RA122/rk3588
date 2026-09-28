import paramiko
import os
import tarfile
import sys
import time

def create_tarball(source_dir, output_filename):
    print(f"打包目录 {source_dir} ...")
    with tarfile.open(output_filename, "w:gz") as tar:
        # 只打包特定的文件夹与模型，忽略不必要的文件
        for item in ["src", "include", "config", "web", "models", "CMakeLists.txt", "README.md"]:
            path = os.path.join(source_dir, item)
            if os.path.exists(path):
                tar.add(path, arcname=f"AILogicBox/{item}")
    print(f"打包完成: {output_filename}")

def ssh_exec(ssh, command, print_output=True, get_pty=False):
    print(f"执行命令: {command}")
    stdin, stdout, stderr = ssh.exec_command(command, get_pty=get_pty)
    out_text = stdout.read().decode('utf-8', errors='ignore')
    err_text = stderr.read().decode('utf-8', errors='ignore')
    exit_status = stdout.channel.recv_exit_status() 
    if print_output:
        if out_text:
            print(out_text)
        if err_text:
            print(f"STDERR:\n{err_text}")
    return exit_status

def main():
    host = "192.168.110.91"
    username = "cat"
    password = "temppwd"
    local_dir = os.path.dirname(os.path.abspath(__file__))
    tar_file = "AILogicBox.tar.gz"

    create_tarball(local_dir, tar_file)

    print(f"正在连接到鲁班猫板卡 ({host})...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, username=username, password=password, timeout=10)
    except Exception as e:
        print(f"连接失败: {e}")
        sys.exit(1)

    print("连接成功！开始上传代码压缩包...")
    sftp = ssh.open_sftp()
    remote_tar = f"/home/{username}/{tar_file}"
    sftp.put(tar_file, remote_tar)
    sftp.close()
    print("上传完成！")

    print("解压代码...")
    ssh_exec(ssh, f"tar -xzf {remote_tar} -C /home/{username}/")

    print("开始编译项目...")
    ssh_exec(ssh, f"mkdir -p /home/{username}/AILogicBox/build")

    print("1. 执行 cmake ...")
    res_cmake = ssh_exec(ssh, f"cd /home/{username}/AILogicBox/build && cmake ..")

    print("2. 执行 make ...")
    res_make = ssh_exec(ssh, f"cd /home/{username}/AILogicBox/build && make -j4")

    print("赋予普通用户读取 NPU 真实负载节点的权限...")
    sudo_prefix = f"echo {password} | sudo -S "
    ssh_exec(ssh, sudo_prefix + "chmod 755 /sys/kernel/debug /sys/kernel/debug/rknpu /sys/kernel/debug/rknpu/load 2>/dev/null || true")

    print("部署和编译流程完成！")
    ssh.close()

    # 清理本地的临时文件
    if os.path.exists(tar_file):
        os.remove(tar_file)

if __name__ == "__main__":
    main()
