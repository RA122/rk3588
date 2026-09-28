import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.110.91', username='cat', password='temppwd', timeout=5)

stdin, stdout, stderr = ssh.exec_command("find / -name 'rknn_api.h' 2>/dev/null")
print(stdout.read().decode())
ssh.close()
