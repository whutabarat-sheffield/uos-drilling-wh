#!/bin/bash
# Deploy test stack with fixed MQTT v2 compatibility

set -e

echo "=== MQTT Test Stack Deployment ==="
echo "This will deploy a test stack with the fixed Paho MQTT v2 compatibility"
echo

# Change to script directory
cd "$(dirname "$0")"

# Build images with fixes
echo "1. Building fixed listener image..."
docker build -t uos-depthest-listener:cpu-test -f ../../Dockerfile ../.. || {
    echo "Failed to build listener image"
    exit 1
}

echo
echo "2. Building publisher image..."
docker build -t uos-publisher:test -f ../../Dockerfile.publisher ../.. || {
    echo "Failed to build publisher image"
    exit 1
}

# Initialize swarm if needed
echo
echo "3. Initializing Docker Swarm..."
if docker info 2>/dev/null | grep -q "Swarm: active"; then
    echo "Swarm already active"
else
    docker swarm init || {
        echo "Failed to initialize swarm"
        exit 1
    }
fi

# Remove existing stack if present
echo
echo "4. Removing existing stack if present..."
docker stack rm mqtt-test 2>/dev/null || true
sleep 5

# Deploy stack
echo
echo "5. Deploying test stack..."
docker stack deploy -c docker-compose.test-swarm.yml mqtt-test || {
    echo "Failed to deploy stack"
    exit 1
}

# Wait for services
echo
echo "6. Waiting for services to start..."
echo "   Waiting for MQTT broker to be ready..."
sleep 15

# Show service status
echo
echo "7. Service status:"
docker service ls | grep -E "(NAME|mqtt-test)" || true

echo
echo "=== Deployment Complete ==="
echo
echo "Monitor logs with:"
echo "  docker service logs -f mqtt-test_uos-depthest-listener-cpu"
echo
echo "Check all services:"
echo "  docker service ls | grep mqtt-test"
echo
echo "Remove stack with:"
echo "  docker stack rm mqtt-test"
echo