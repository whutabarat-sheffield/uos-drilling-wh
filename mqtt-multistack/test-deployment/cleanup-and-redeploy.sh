#!/bin/bash
# Cleanup and redeploy the test stack

echo "Cleaning up old deployment..."
docker stack rm mqtt-test
sleep 5

echo "Removing old containers..."
docker rm -f $(docker ps -aq -f name=mqtt-test) 2>/dev/null || true

echo "Redeploying..."
./deploy-test-stack.sh