# MQTT Publisher Stress Testing Guide

## Overview

The UOS Drilling platform now includes a high-performance stress testing mode that can publish up to 1000 signals per second (3000 MQTT messages/second) to evaluate system performance under load.

## Features

- **Concurrent Publishing**: Uses multiple threads and MQTT clients for parallel publishing
- **Performance Metrics**: Real-time monitoring of throughput, latency, and success rates
- **Configurable Load**: Adjust target rate, duration, and concurrency levels
- **Pre-cached Data**: Loads all test data into memory before testing to minimize I/O
- **Optimized MQTT Settings**: Increased inflight messages and queue sizes

## Usage

### Basic Stress Test

Run a 60-second test at 1000 signals/second:

```bash
python uos_publish_wrapper.py test_data --stress-test --rate 1000 --duration 60
```

### Docker Usage

```bash
# Build the image
./build-publish.sh

# Run stress test
docker run --rm uos-publish:latest \
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

## Performance Tuning

### MQTT Client Settings

The stress test publisher optimizes MQTT client settings:

```python
client.max_inflight_messages_set(100)  # Default: 20
client.max_queued_messages_set(1000)   # Default: 100
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
   - Increase max_connections
   - Tune message buffer sizes

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