import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.110.91', username='cat', password='temppwd', timeout=5)

stdin, stdout, stderr = ssh.exec_command("tail -n 100 /home/cat/AILogicBox/ai_box.log")
print("STDOUT:")
print(stdout.read().decode())
print("STDERR:")
print(stderr.read().decode())
ssh.close()
