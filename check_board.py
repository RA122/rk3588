import paramiko
import sys

def run_ssh(cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect('192.168.110.91', username='cat', password='temppwd', timeout=5)
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print("STDOUT:")
        print(stdout.read().decode())
        print("STDERR:")
        print(stderr.read().decode())
        ssh.close()
    except Exception as e:
        print(f"Error: {e}")

run_ssh("rm -f /home/cat/AILogicBox/build/alerts/*.jpg")
print("Cleared old alerts.")
