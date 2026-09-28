import paramiko
import json

HOST = '192.168.110.91'
USER = 'cat'
PASS = 'temppwd'

# 在板子上跑诊断: 列出 models/ 下的 .rknn 及其同名 sidecar (.json/.labels/.txt/.names),
# 再 curl 本机的 /api/models 和 /api/model_classes?name=... 验证后端路径是否正确.

REMOTE_CMDS = r'''
echo "=== CWD ==="
pwd

echo
echo "=== ../models/ 下的 .rknn 文件 (filename + 同名 sidecar 是否存在) ==="
cd ../models 2>/dev/null || { echo "无法进入 ../models (CWD=${pwd})"; exit 1; }
for rknn in *.rknn; do
    [ -e "$rknn" ] || continue
    base="${rknn%.rknn}"
    json="$base.json"; labels="$base.labels"; txt="$base.txt"; names="$base.names"
    hits=""
    [ -f "$json"  ] && hits="$hits $json"
    [ -f "$labels" ] && hits="$hits $labels"
    [ -f "$txt"   ] && hits="$hits $txt"
    [ -f "$names" ] && hits="$hits $names"
    if [ -z "$hits" ]; then
        echo "[NO SIDECAR] $rknn   (期望同目录存在 $base.json / .labels / .txt / .names 任一)"
    else
        echo "[OK]         $rknn   sidecar:$hits"
    fi
done

echo
echo "=== /api/models (HTTP) ==="
curl -sS -m 3 http://127.0.0.1:8080/api/models 2>&1 || echo "curl 失败 (端口/服务可能没起)"
echo
echo
echo "=== /api/model_classes?name=xxx (HTTP, 对每个 .rknn 都试一次) ==="
PORT="${PORT:-8080}"
for rknn in *.rknn; do
    [ -e "$rknn" ] || continue
    echo "--- $rknn ---"
    curl -sS -m 3 "http://127.0.0.1:${PORT}/api/model_classes?name=${rknn}" 2>&1
    echo
done
'''

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS, timeout=5)

# 直接执行上面的多行 shell
stdin, stdout, stderr = ssh.exec_command("bash -s", timeout=30)
stdin.write(REMOTE_CMDS)
stdin.channel.shutdown_write()

out = stdout.read().decode(errors='replace')
err = stderr.read().decode(errors='replace')

print("===== STDOUT =====")
print(out)
print("===== STDERR =====")
print(err)

ssh.close()