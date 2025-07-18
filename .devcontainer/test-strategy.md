# Devcontainer Testing Strategy for v0.2.6 Compatibility

## Testing Objectives

Ensure the devcontainer:
1. Starts successfully in multiple network configurations
2. Supports stress testing capabilities
3. Integrates with Portainer deployments
4. Provides proper development environment for v0.2.6 features

## Test Scenarios

### 1. Network Connectivity Tests

#### Test 1.1: No MQTT Network Available
**Setup:**
```bash
# Ensure no MQTT networks exist
docker network prune -f
```

**Expected Result:**
- Devcontainer starts with bridge network
- Warning message about MQTT not available
- Can still develop/test code

#### Test 1.2: Standard Docker Compose Network
**Setup:**
```bash
cd mqtt-multistack/mqtt-broker
docker-compose up -d
# Creates 'toolbox-network'
```

**Expected Result:**
- Devcontainer auto-detects network
- Can connect to mqtt-broker
- mosquitto_sub/pub commands work

#### Test 1.3: Portainer Multi-stack Network
**Setup:**
```bash
# Simulate Portainer network naming
docker network create mqtt-broker_toolbox-network
docker run -d --name mqtt-broker --network mqtt-broker_toolbox-network eclipse-mosquitto
```

**Expected Result:**
- Devcontainer detects Portainer network
- Connects successfully
- No manual configuration needed

#### Test 1.4: Custom Network Name
**Setup:**
```bash
docker network create custom-mqtt-network
export DETECTED_NETWORK=custom-mqtt-network
```

**Expected Result:**
- Devcontainer uses environment variable
- Connects to custom network

### 2. Tool Availability Tests

#### Test 2.1: MQTT Client Tools
```bash
# Inside devcontainer
which mosquitto_pub
which mosquitto_sub
mosquitto_pub -h mqtt-broker -t test -m "Hello"
mosquitto_sub -h mqtt-broker -t test -C 1
```

#### Test 2.2: Stress Testing Tools
```bash
# Verify stress test dependencies
python -c "import aiomqtt; print('aiomqtt available')"
python -c "import asyncio; print('asyncio available')"

# Run basic stress test
cd /workspaces/*/abyss
python src/abyss/run/uos_publish_wrapper.py --help | grep stress
```

#### Test 2.3: Docker Access
```bash
# Verify Docker socket mount
docker ps
docker network ls
docker run --rm hello-world
```

#### Test 2.4: Performance Monitoring
```bash
# Check monitoring tools
htop --version
curl --version
jq --version
```

### 3. Port Forwarding Tests

#### Test 3.1: MQTT Ports
```bash
# From host machine
nc -zv localhost 1883  # MQTT
nc -zv localhost 9001  # WebSocket
nc -zv localhost 8883  # TLS
```

#### Test 3.2: Monitoring Ports
```bash
# When stress test is running
curl http://localhost:9090/metrics  # Prometheus metrics
```

### 4. Environment Variable Tests

#### Test 4.1: Default Values
```bash
# Inside devcontainer
echo $MQTT_BROKER_HOST  # Should be 'mqtt-broker'
echo $MQTT_BROKER_PORT  # Should be '1883'
echo $STRESS_TEST_ENABLED  # Should be 'true'
```

#### Test 4.2: Custom Values
```bash
# Set before opening devcontainer
export MQTT_BROKER_HOST=custom-broker
export MQTT_BROKER_PORT=1884

# Inside devcontainer
echo $MQTT_BROKER_HOST  # Should be 'custom-broker'
```

### 5. Cache Management Tests

#### Test 5.1: Cache Persistence
```bash
# Install package
pip install numpy

# Rebuild container
# Verify numpy still in cache
pip install numpy  # Should be fast
```

#### Test 5.2: Cache Invalidation
```bash
# Change CACHE_VERSION
export CACHE_VERSION=v2

# Rebuild container
# Verify clean cache
ls -la ~/.cache/pip/  # Should be empty/new
```

### 6. Integration Tests

#### Test 6.1: Full Stress Test Workflow
```bash
# Inside devcontainer
cd /workspaces/*/abyss

# Run stress test
python src/abyss/run/uos_publish_wrapper.py \
    src/abyss/test_data \
    --stress-test \
    --rate 100 \
    --duration 10 \
    --concurrent-publishers 5
```

#### Test 6.2: Portainer API Access
```bash
# If Portainer credentials provided
export PORTAINER_URL=http://portainer:9000
export PORTAINER_API_KEY=your-key

# Test API access
curl -H "X-API-Key: $PORTAINER_API_KEY" $PORTAINER_URL/api/stacks
```

## Test Automation Script

Create `.devcontainer/test-devcontainer.sh`:

```bash
#!/bin/bash
set -e

echo "=== Devcontainer Test Suite ==="

# Test 1: Network detection
echo "Testing network detection..."
NETWORKS=$(docker network ls --format '{{.Name}}' | grep -E '(mqtt|toolbox)' || true)
echo "Found networks: $NETWORKS"

# Test 2: MQTT connectivity
echo "Testing MQTT connectivity..."
if command -v mosquitto_pub &> /dev/null; then
    echo "✓ MQTT clients installed"
    if mosquitto_pub -h ${MQTT_BROKER_HOST:-mqtt-broker} -t test -m "test" -W 3; then
        echo "✓ MQTT broker reachable"
    else
        echo "✗ MQTT broker not reachable"
    fi
else
    echo "✗ MQTT clients not installed"
fi

# Test 3: Python dependencies
echo "Testing Python dependencies..."
python -c "
import sys
try:
    import aiomqtt
    print('✓ aiomqtt available')
except ImportError:
    print('✗ aiomqtt not available')
    sys.exit(1)
"

# Test 4: Docker access
echo "Testing Docker access..."
if docker ps &> /dev/null; then
    echo "✓ Docker socket accessible"
else
    echo "✗ Docker socket not accessible"
fi

# Test 5: Environment variables
echo "Testing environment variables..."
[[ -n "$STRESS_TEST_ENABLED" ]] && echo "✓ STRESS_TEST_ENABLED set" || echo "✗ STRESS_TEST_ENABLED not set"
[[ -n "$MQTT_BROKER_HOST" ]] && echo "✓ MQTT_BROKER_HOST set" || echo "✗ MQTT_BROKER_HOST not set"

# Test 6: Port accessibility
echo "Testing port forwarding..."
for port in 1883 9001; do
    if nc -z localhost $port 2>/dev/null; then
        echo "✓ Port $port accessible"
    else
        echo "✗ Port $port not accessible"
    fi
done

echo "=== Test Suite Complete ==="
```

## Success Criteria

1. **All network scenarios**: Devcontainer starts successfully
2. **Tool availability**: All required tools installed and functional
3. **MQTT connectivity**: Can connect when broker available, graceful failure when not
4. **Performance**: Container starts in <30 seconds
5. **Stress testing**: Can run stress tests at 1000+ msg/sec
6. **No breaking changes**: Existing workflows continue to work

## Rollback Plan

If issues arise:
1. Restore original devcontainer.json from backup
2. Document specific failure scenarios
3. Create minimal fix for critical issues only
4. Plan phased migration for complex changes