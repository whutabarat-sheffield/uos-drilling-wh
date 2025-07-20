#!/bin/bash
# Test locally with docker-compose (no swarm)

set -e

echo "=== MQTT Local Test Deployment ==="
echo "Testing with docker-compose (no swarm required)"
echo

# Change to script directory
cd "$(dirname "$0")"

# Build images
echo "1. Building images..."
docker-compose -f docker-compose.local.yml build || {
    echo "Failed to build images"
    exit 1
}

# Stop existing containers
echo
echo "2. Stopping existing containers..."
docker-compose -f docker-compose.local.yml down

# Start services
echo
echo "3. Starting services..."
docker-compose -f docker-compose.local.yml up -d || {
    echo "Failed to start services"
    exit 1
}

# Wait for services
echo
echo "4. Waiting for services to be ready..."
sleep 10

# Show status
echo
echo "5. Service status:"
docker-compose -f docker-compose.local.yml ps

echo
echo "=== Deployment Complete ==="
echo
echo "Monitor logs with:"
echo "  docker-compose -f docker-compose.local.yml logs -f uos-depthest-listener-cpu"
echo
echo "Stop all services:"
echo "  docker-compose -f docker-compose.local.yml down"
echo