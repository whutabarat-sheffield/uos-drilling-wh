# MQTT Test Deployment

This directory contains a Docker Swarm test stack for validating the Paho MQTT v2 compatibility fixes.

## What This Tests

1. **Fixed on_publish callback** - Verifies the 5-parameter callback signature works with Paho MQTT v2
2. **Publisher health recovery** - Tests automatic reconnection if publisher fails
3. **End-to-end data flow** - Publisher → Broker → Listener → Processing → Results

## Components

- **mqtt-broker**: Eclipse Mosquitto MQTT broker
- **uos-depthest-listener-cpu**: Fixed depth estimation listener (CPU-only)
- **uos-publisher-json**: Publishes test data from `simple-tracking/data`

## Quick Start

### Option 1: Docker Swarm (Production-like)

```bash
# Deploy the test stack
./deploy-test-stack.sh

# Monitor the listener logs (watch for successful publishes)
docker service logs -f mqtt-test_uos-depthest-listener-cpu

# Check service status
docker service ls | grep mqtt-test

# Remove the stack when done
docker stack rm mqtt-test
```

### Option 2: Local Docker Compose (Simpler)

```bash
# Run local test (no swarm needed)
./test-local.sh

# Monitor the listener logs
docker-compose -f docker-compose.local.yml logs -f uos-depthest-listener-cpu

# Stop all services
docker-compose -f docker-compose.local.yml down
```

## Expected Results

With the fixes applied, you should see:
- ✅ No TypeError on `on_publish` callback
- ✅ Processing rate > 0 msg/s
- ✅ Successful depth estimation results published
- ✅ Publisher remains connected and healthy

## Troubleshooting

1. **Build fails**: Ensure you're in the `test-deployment` directory
2. **Services don't start**: Check `docker service ps mqtt-test_<service>`
3. **No messages**: Verify data files exist in `../simple-tracking/data/data_20250326`

## Configuration

- Broker config: `mosquitto.conf`
- MQTT settings: `../../abyss/src/abyss/run/config/mqtt_conf_docker.yaml`
- Test data: `../simple-tracking/data/data_20250326/*.json`