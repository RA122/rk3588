import paramiko
import time

host = "192.168.110.91"
username = "cat"
password = "temppwd"

print(f"Connecting to {host}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(host, username=username, password=password, timeout=10)
    
    # 检查是否已经在运行，如果有则杀掉旧进程
    ssh.exec_command("pkill -f ai_box")
    time.sleep(1)
    
    # 后台启动 ai_box
    print("Starting ai_box on the board...")
    # 确保当前目录是 build，因为程序内部用了 "../config/config.json" 这种相对路径
    cmd = "cd /home/cat/AILogicBox/build && nohup ./ai_box > /home/cat/AILogicBox/ai_box.log 2>&1 &"
    ssh.exec_command(cmd)

    # 等待一两秒，读取一下启动日志
    time.sleep(2)
    stdin, stdout, stderr = ssh.exec_command("cat /home/cat/AILogicBox/ai_box.log")
    print("--- Startup Log ---")
    print(stdout.read().decode('utf-8'))
    
    ssh.close()
    print("Done.")
except Exception as e:
    print(f"Error: {e}")
