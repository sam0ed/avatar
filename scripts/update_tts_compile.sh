#!/bin/bash
# Temporary script to update TTS config with COMPILE=1 on the running instance.
# Run via: scp to instance, then bash it.

python3 -c '
with open("/etc/supervisor/conf.d/avatar.conf") as f:
    content = f.read()

old = """[program:tts]
command=bash start_server.sh
directory=/app
priority=20
autostart=true
autorestart=true
startretries=3
startsecs=10"""

new = """[program:tts]
command=bash start_server.sh
directory=/app
environment=COMPILE="1"
priority=20
autostart=true
autorestart=true
startretries=3
startsecs=30"""

content = content.replace(old, new)
with open("/etc/supervisor/conf.d/avatar.conf", "w") as f:
    f.write(content)
print("Config updated: COMPILE=1 added to TTS")
'

# Reload supervisor and restart TTS
supervisorctl reread
supervisorctl update
supervisorctl restart tts

echo "TTS restarting with COMPILE=1..."
echo "Waiting for TTS health..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8080/v1/health > /dev/null 2>&1; then
        echo "TTS health OK after ${i}s"
        break
    fi
    sleep 1
done

echo "Sending warmup request (torch.compile first call)..."
WARMUP_START=$(date +%s)
curl -sf -X POST http://localhost:8080/v1/tts \
    -H 'Content-Type: application/json' \
    -d '{"text": "Hello world.", "streaming": false}' \
    -o /dev/null || echo "Warmup failed (non-fatal)"
WARMUP_END=$(date +%s)
echo "Warmup completed in $((WARMUP_END - WARMUP_START))s"

echo ""
echo "Verifying all services..."
supervisorctl status
