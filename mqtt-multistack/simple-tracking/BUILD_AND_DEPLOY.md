# Build and Deploy Instructions for Portainer

## Quick Start (Recommended)

### 1. Build the Custom Image

From the `simple-tracking` directory:

```bash
# Build the image with signal tracking
docker build -f Dockerfile.publisher-tracked -t uos-publish-tracked:latest ../..

# Test locally
docker run --rm uos-publish-tracked:latest python /app/uos_publish_json.py --help
```

### 2. Push to Registry (if using remote Portainer)

```bash
# Tag for your registry
docker tag uos-publish-tracked:latest your-registry/uos-publish-tracked:latest

# Push
docker push your-registry/uos-publish-tracked:latest
```

### 3. Deploy in Portainer

Use this simplified compose file:

```yaml
version: '3.8'

services:
  publisher:
    image: uos-publish-tracked:latest  # or your-registry/uos-publish-tracked:latest
    container_name: signal-publisher
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - tracking:/tracking
    networks:
      - mqtt-broker_toolbox-network

  signal-monitor:
    image: python:3.10.16-slim
    container_name: signal-monitor
    command: >
      sh -c "
      pip install paho-mqtt >/dev/null 2>&1;
      python -c \"
import csv, json, time, paho.mqtt.client as mqtt
def on_message(c, u, m):
    try:
        d = json.loads(m.payload)
        sid = d.get('_signal_id')
        if sid:
            with open('/tracking/received_signals.csv', 'a') as f:
                csv.writer(f).writerow([sid, time.time(), m.topic])
            print(f'Tracked: {sid}')
    except: pass
client = mqtt.Client()
client.on_message = on_message
client.connect('mqtt-broker', 1883)
client.subscribe('#')
print('Monitor started')
client.loop_forever()
      \""
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - tracking:/tracking
    networks:
      - mqtt-broker_toolbox-network

  stats:
    image: busybox
    container_name: signal-stats
    command: >
      sh -c "
      while true; do
        clear;
        echo '=== Signal Tracking ===';
        [ -f /tracking/sent_signals.csv ] && echo \"Sent: $$(wc -l < /tracking/sent_signals.csv)\" || echo 'Sent: 0';
        [ -f /tracking/received_signals.csv ] && echo \"Received: $$(wc -l < /tracking/received_signals.csv)\" || echo 'Received: 0';
        echo '====================';
        sleep 5;
      done"
    volumes:
      - tracking:/tracking:ro
    networks:
      - mqtt-broker_toolbox-network

volumes:
  tracking:

networks:
  mqtt-broker_toolbox-network:
    external: true
```

## Alternative: Use Existing Image

If you can't build a custom image, you need to:

1. Ensure test data exists in the container
2. Mount the modified script

See `docker-compose.portainer-working.yml` for an example that creates test data on startup.

## Verification

After deployment:

```bash
# Check publisher logs
docker logs signal-publisher

# Check monitor logs  
docker logs signal-monitor

# View statistics
docker logs -f signal-stats
```