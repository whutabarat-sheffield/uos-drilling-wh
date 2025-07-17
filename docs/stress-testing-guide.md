# MQTT Publisher Stress Testing Guide

## Overview

The UOS Drilling platform includes multiple high-performance stress testing modes designed to push MQTT publishing to its limits. While the target is 1000 signals/second, actual performance depends on system resources and MQTT broker capacity.

## Performance Modes

### 1. **Threading Mode** (Default)
- Uses ThreadPoolExecutor with multiple MQTT clients
- Achieves ~80-100 signals/second with default settings
- Good balance of performance and compatibility

### 2. **Async Mode** (--async)
- Uses asyncio with aiomqtt for asynchronous publishing
- Potential for higher throughput with proper tuning
- Requires aiomqtt library installation

### 3. **Batch Mode** (--batch-mode)
- Groups multiple messages for efficient publishing
- Reduces per-message overhead
- Configurable batch sizes

## Features

- **Concurrent Publishing**: Multiple threads/coroutines for parallel publishing
- **Connection Pooling**: Multiple MQTT clients to distribute load
- **Performance Metrics**: Real-time monitoring of throughput, latency, and success rates
- **Pre-cached Data**: Eliminates I/O bottlenecks during testing
- **Optimized MQTT Settings**: Tuned for maximum throughput
- **Network Auto-detection**: Works with various Docker deployments

## Usage

### Quick Start with Helper Script

The easiest way to run stress tests is using the `run-stress-test.sh` script:

```bash
# Auto-detect network and run default test (threading mode)
./run-stress-test.sh

# Run async mode for potentially higher performance
./run-stress-test.sh --async -r 1000 -p 50

# Build image and run test
./run-stress-test.sh -b

# Use specific network
./run-stress-test.sh -N mqtt-broker_toolbox-network

# Run in standalone mode (host network)
./run-stress-test.sh --standalone

# Maximum performance attempt
./run-stress-test.sh --async -r 1000 -p 100 --no-sleep
```

### Basic Stress Test

Run a 60-second test at 1000 signals/second:

```bash
python uos_publish_wrapper.py test_data --stress-test --rate 1000 --duration 60
```

### Docker Usage

```bash
# Build the image
./build-publish.sh

# Run stress test with auto-detected network
./run-stress-test.sh

# Manual Docker run
docker run --rm --network mqtt-broker_toolbox-network \
  uos-publish-json:latest \
  python uos_publish_wrapper.py test_data \
  --stress-test \
  --rate 1000 \
  --duration 60 \
  --concurrent-publishers 10
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--stress-test` | Enable stress testing mode | False |
| `--rate` | Target signals per second | 1000 |
| `--duration` | Test duration in seconds | 60 |
| `--concurrent-publishers` | Number of concurrent threads | 10 |
| `--batch-size` | Signals per batch | 1 |
| `--no-sleep` | Disable rate limiting | False |

## Network Configuration

The stress test script supports multiple network configurations to work with different deployment scenarios:

### Auto-Detection (Default)

The script automatically detects MQTT networks in this order:
1. `mqtt-broker_toolbox-network` (Portainer multi-stack)
2. `toolbox-network` (Direct docker-compose)
3. Falls back to host network if no MQTT network found

```bash
# Auto-detect network
./run-stress-test.sh
```

### Manual Network Selection

```bash
# Use specific network
./run-stress-test.sh -N mqtt-broker_toolbox-network

# Use host network (standalone mode)
./run-stress-test.sh --standalone

# Create temporary network (for isolated testing)
./run-stress-test.sh --create-network
```

### Network Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Auto-detect | Finds MQTT broker network automatically | Portainer/Docker Compose deployments |
| Specific (`-N`) | Use named network | Known network configuration |
| Standalone | Use host network | Local MQTT broker on host |
| Create Network | Create temporary isolated network | Testing without broker |

## Performance Bottlenecks and Analysis

### Current Performance Limits

Based on testing, the system achieves:
- **Threading mode**: ~80-100 signals/second
- **Async mode**: Potentially higher with proper tuning
- **Batch mode**: Reduces overhead but similar limits

### Identified Bottlenecks

1. **MQTT Broker Capacity**: The broker may be the limiting factor
2. **Python GIL**: Threading is limited by Global Interpreter Lock
3. **Network Latency**: Each publish requires network round-trip
4. **Message Size**: Large JSON payloads impact throughput

### Optimization Strategies

1. **Use Multiple Processes**: Instead of threads, use multiprocessing
2. **Optimize Broker**: Tune mosquitto for high throughput
3. **Reduce Message Size**: Compress or minimize JSON payloads
4. **Use MQTT 5.0**: Better performance with newer protocol
5. **Batch Publishing**: Group messages to reduce overhead

## Performance Tuning

### MQTT Client Settings

The stress test publisher optimizes MQTT client settings:

```python
# Threading mode
client.max_inflight_messages_set(100)  # Default: 20
client.max_queued_messages_set(1000)   # Default: 100

# Async mode (even more aggressive)
client = aiomqtt.Client(
    max_inflight_messages=200,
    max_queued_messages=5000,
    protocol=aiomqtt.ProtocolVersion.V5
)
```

### System Tuning

For best performance, consider:

1. **File Descriptors**: Increase ulimit
   ```bash
   ulimit -n 65536
   ```

2. **Network Buffers**: Tune TCP settings
   ```bash
   sudo sysctl -w net.core.rmem_max=134217728
   sudo sysctl -w net.core.wmem_max=134217728
   ```

3. **MQTT Broker**: Ensure mosquitto/broker can handle load
   - Increase max_connections in mosquitto.conf
   - Set persistence false for testing
   - Increase max_inflight_messages
   - Tune system limits:
   ```bash
   # /etc/mosquitto/mosquitto.conf
   max_connections -1
   max_inflight_messages 1000
   max_queued_messages 10000
   persistence false
   ```

## Metrics Output

The stress test provides real-time metrics every 5 seconds:

```
============================================================
Stress Test Metrics - Elapsed: 10.2s
============================================================
Messages Sent:     30,456
Messages Failed:   12
Success Rate:      99.96%
Signals/Second:    996.7 (target: 1000)
Messages/Second:   2,990.2
Throughput:        24.56 Mbps
Avg Latency:       0.82 ms
Min/Max Latency:   0.45 / 2.31 ms
============================================================
```

## Interpreting Results

### Key Metrics

1. **Signals/Second**: Should approach target rate
2. **Success Rate**: Should be >99% under normal conditions
3. **Latency**: Lower is better, <5ms is excellent
4. **Throughput**: Depends on message size

### Common Issues

1. **Low Signal Rate**: 
   - Increase concurrent publishers
   - Check network/broker capacity
   - Use `--no-sleep` for maximum throughput

2. **High Failure Rate**:
   - Broker overloaded
   - Network issues
   - Client queue overflow

3. **High Latency**:
   - Broker processing bottleneck
   - Network congestion
   - Insufficient concurrent publishers

## Example Test Scenarios

### Light Load Test (100 signals/sec)
```bash
python uos_publish_wrapper.py test_data \
  --stress-test \
  --rate 100 \
  --duration 300 \
  --concurrent-publishers 2
```

### Medium Load Test (500 signals/sec)
```bash
python uos_publish_wrapper.py test_data \
  --stress-test \
  --rate 500 \
  --duration 120 \
  --concurrent-publishers 5
```

### Maximum Load Test (1000+ signals/sec)
```bash
python uos_publish_wrapper.py test_data \
  --stress-test \
  --rate 1000 \
  --duration 60 \
  --concurrent-publishers 20 \
  --no-sleep
```

## Monitoring During Tests

### MQTT Broker
Monitor mosquitto logs:
```bash
docker logs -f mosquitto
```

### System Resources
```bash
# CPU and Memory
htop

# Network
iftop

# Disk I/O
iotop
```

### Application Logs
The stress test logs important events:
- Connection status
- Worker thread creation
- Error conditions
- Final summary

## Best Practices

1. **Warm-up**: Run a short test first to establish connections
2. **Incremental Testing**: Start with low rates and increase gradually
3. **Monitor Resources**: Watch CPU, memory, and network usage
4. **Test Duration**: Run for at least 60 seconds for meaningful results
5. **Multiple Runs**: Average results from multiple test runs

## Troubleshooting

### "Connection Refused"
- Check broker is running
- Verify host/port configuration
- Check firewall rules

### "Queue Full" Errors
- Reduce publish rate
- Increase queue sizes
- Add more concurrent publishers

### Low Achievement Rate
- System may be at capacity
- Try increasing concurrent publishers
- Check for network bottlenecks
- Monitor broker performance

### Network Errors

#### "network not found"
```bash
# List available networks
docker network ls

# Use auto-detection
./run-stress-test.sh

# Or use host network
./run-stress-test.sh --standalone
```

#### "MQTT broker not accessible"
```bash
# Check if broker is running
docker ps | grep mqtt

# Test with host network
./run-stress-test.sh --standalone

# Verify broker hostname in config
# Docker: mqtt-broker
# Local: localhost
```

#### MQTT Client Version Errors
The stress test handles both old and new paho-mqtt versions automatically. If you see version errors:
```bash
# Rebuild the image
./build-publish.sh

# Run test
./run-stress-test.sh
```

## Achieving 1000 Signals/Second Target

### Current Status

The current implementation achieves approximately 80-100 signals/second, which is about 10% of the target. This is due to several fundamental limitations:

1. **Python GIL**: Prevents true parallel execution in threads
2. **MQTT Protocol Overhead**: Each message requires acknowledgment
3. **Network Latency**: Docker networking adds overhead
4. **Large Payload Size**: JSON messages are several KB each

### Recommendations for 1000 Signals/Second

To achieve the target rate, consider:

1. **Use Multiple Containers**: Run 10-20 publisher containers in parallel
   ```bash
   # Run 10 parallel publishers, each doing 100 signals/sec
   for i in {1..10}; do
     docker run -d --name publisher-$i uos-publish-json:latest \
       python uos_publish_wrapper.py test_data \
       --stress-test --rate 100 --concurrent-publishers 5
   done
   ```

2. **Switch to Compiled Language**: Use Go, Rust, or C++ for publishing
   - These languages don't have GIL limitations
   - Can achieve true parallel execution
   - Much lower overhead per message

3. **Use MQTT Clustering**: Distribute load across multiple brokers
   - Set up multiple mosquitto instances
   - Use load balancer to distribute connections
   - Each broker handles portion of the load

4. **Optimize Message Format**: 
   - Use binary format (MessagePack, Protocol Buffers)
   - Compress JSON with gzip
   - Send only changed data (deltas)

5. **Hardware Acceleration**:
   - Use dedicated network cards
   - Enable jumbo frames
   - Use kernel bypass networking (DPDK)

### Realistic Expectations

For Python-based publishing with standard MQTT:
- **Single Publisher**: 80-100 signals/second
- **Multiple Publishers**: 500-800 signals/second (with 10 containers)
- **With Optimizations**: Up to 1000 signals/second possible

For production 1000+ signals/second, consider:
- Industrial IoT platforms (Apache Kafka, RabbitMQ)
- Time-series databases with native ingestion
- Custom protocol over UDP for lower overhead

## Integration with CI/CD

Add stress testing to your pipeline:

```yaml
stress-test:
  stage: performance
  script:
    - docker run uos-publish:latest python uos_publish_wrapper.py test_data \
        --stress-test --rate 500 --duration 30
  only:
    - schedules
```