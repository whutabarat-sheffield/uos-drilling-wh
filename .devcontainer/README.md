# UOS Drilling Development Container v0.2.6

This devcontainer provides a complete development environment for the UOS Drilling project with MQTT integration and stress testing capabilities.

## Quick Start

### Option 1: Simple Setup (Default)
1. Open the project in VS Code
2. When prompted, click "Reopen in Container"
3. The container will start with localhost MQTT configuration

### Option 2: With MQTT Network
1. Ensure the MQTT broker is running:
   ```bash
   cd mqtt-multistack/mqtt-broker
   docker-compose up -d
   ```
2. Use the networked devcontainer:
   - In VS Code: Select `.devcontainer/with-mqtt/devcontainer.json` when prompted
   - Or copy it to the root: `cp .devcontainer/with-mqtt/devcontainer.json .devcontainer/devcontainer.json`
3. Reopen in container

## Configuration Options

### Environment Variables
Set these before opening the devcontainer:

```bash
# MQTT Configuration
export MQTT_BROKER_HOST=your-broker    # Default: localhost
export MQTT_BROKER_PORT=1883           # Default: 1883
export STRESS_TEST_ENABLED=true        # Default: true

# Cache Management
export CACHE_VERSION=v2                # Increment to clear pip cache
```

### Network Setup

The devcontainer attempts to create the MQTT network automatically. If you need to create it manually:

```bash
docker network create mqtt-broker_toolbox-network
```

## Features

### Included Tools
- **MQTT Clients**: mosquitto_pub, mosquitto_sub
- **Stress Testing**: aiomqtt, prometheus-client
- **Performance Monitoring**: htop, Docker CLI access
- **Development**: Python 3.10, git, build tools

### Port Forwarding
- `1883` - MQTT standard port
- `9001` - MQTT WebSocket port  
- `8883` - MQTT TLS port (if configured)
- `9090` - Prometheus metrics (stress testing)

### Volume Caching
Persistent caches for faster rebuilds:
- pip packages
- transformers models
- matplotlib config
- PyTorch models

## Testing MQTT Connectivity

### Check Connection
```bash
# Inside the devcontainer
mosquitto_pub -h $MQTT_BROKER_HOST -p $MQTT_BROKER_PORT -t test -m "Hello"
mosquitto_sub -h $MQTT_BROKER_HOST -p $MQTT_BROKER_PORT -t test -C 1
```

### Run Stress Test
```bash
cd /workspaces/uos-drilling-wh/abyss
python src/abyss/run/uos_publish_wrapper.py \
    src/abyss/test_data \
    --stress-test \
    --rate 1000 \
    --duration 60 \
    --concurrent-publishers 10
```

## Troubleshooting

### MQTT Connection Failed
If you see "MQTT broker not reachable":
1. Check if the broker is running: `docker ps | grep mqtt`
2. Verify network exists: `docker network ls | grep mqtt`
3. Try localhost instead: `export MQTT_BROKER_HOST=localhost`

### Performance Issues
1. Increase Docker resources in Docker Desktop settings
2. Use the simple devcontainer (not Docker Compose) for better performance
3. Disable file watchers for large directories

### Cache Issues
Clear all caches:
```bash
docker volume rm pip-cache-v1 transformers-cache matplotlib-cache torch-cache
```

## Architecture Decision

We chose a **simple, fail-safe approach** over complex dynamic network detection:
- The devcontainer works with or without MQTT
- Uses localhost by default with port forwarding
- Provides clear feedback about connectivity status
- Offers Docker Compose alternative for guaranteed network access

This ensures developers can work immediately without complex setup while still supporting full MQTT integration when needed.