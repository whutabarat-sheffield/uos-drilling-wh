#!/bin/bash
set -e

echo "=== Devcontainer Test Suite v0.2.6 ==="
echo "Running comprehensive tests for devcontainer functionality"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test result counters
PASSED=0
FAILED=0
WARNINGS=0

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    local is_warning="${3:-false}"
    
    echo -n "Testing $test_name... "
    if eval "$test_command" &> /dev/null; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
    else
        if [ "$is_warning" = "true" ]; then
            echo -e "${YELLOW}⚠ WARNING${NC}"
            ((WARNINGS++))
        else
            echo -e "${RED}✗ FAILED${NC}"
            ((FAILED++))
        fi
    fi
}

echo "=== Environment Information ==="
echo "Container OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
echo "Python Version: $(python --version 2>&1)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo ""

echo "=== Test 1: Network Detection ==="
NETWORKS=$(docker network ls --format '{{.Name}}' 2>/dev/null | grep -E '(mqtt|toolbox)' || true)
if [ -n "$NETWORKS" ]; then
    echo "Found networks: $NETWORKS"
    run_test "Docker network availability" "[ -n '$NETWORKS' ]"
else
    echo "No MQTT-related networks found"
    run_test "Docker network availability" "false" "true"
fi

echo ""
echo "=== Test 2: MQTT Tools ==="
run_test "mosquitto_pub installed" "command -v mosquitto_pub"
run_test "mosquitto_sub installed" "command -v mosquitto_sub"

echo ""
echo "=== Test 3: MQTT Connectivity ==="
MQTT_HOST=${MQTT_BROKER_HOST:-localhost}
MQTT_PORT=${MQTT_BROKER_PORT:-1883}
echo "MQTT Configuration: $MQTT_HOST:$MQTT_PORT"
run_test "MQTT broker connectivity" "timeout 2 mosquitto_pub -h $MQTT_HOST -p $MQTT_PORT -t test/devcontainer -m 'test' 2>/dev/null" "true"

echo ""
echo "=== Test 4: Python Dependencies ==="
run_test "aiomqtt module" "python -c 'import aiomqtt'"
run_test "aiofiles module" "python -c 'import aiofiles'"
run_test "prometheus_client module" "python -c 'import prometheus_client'"
run_test "asyncio module" "python -c 'import asyncio'"

echo ""
echo "=== Test 5: Development Tools ==="
run_test "git installed" "command -v git"
run_test "build-essential" "command -v gcc"
run_test "vim-tiny installed" "command -v vim.tiny"
run_test "htop installed" "command -v htop"
run_test "curl installed" "command -v curl"
run_test "jq installed" "command -v jq"
run_test "netcat installed" "command -v nc"

echo ""
echo "=== Test 6: Docker Access ==="
run_test "Docker CLI available" "command -v docker"
run_test "Docker socket accessible" "docker ps" "true"

echo ""
echo "=== Test 7: Environment Variables ==="
run_test "PYTHONUNBUFFERED set" "[ -n '$PYTHONUNBUFFERED' ]"
run_test "LOG_LEVEL set" "[ -n '$LOG_LEVEL' ]"
run_test "STRESS_TEST_ENABLED set" "[ -n '$STRESS_TEST_ENABLED' ]"
run_test "MQTT_BROKER_HOST set" "[ -n '$MQTT_BROKER_HOST' ]"
run_test "MQTT_BROKER_PORT set" "[ -n '$MQTT_BROKER_PORT' ]"

echo ""
echo "=== Test 8: Cache Directories ==="
run_test "pip cache directory" "[ -d ~/.cache/pip ]"
run_test "matplotlib cache directory" "[ -d ~/.cache/matplotlib ]"
run_test "transformers cache directory" "[ -d ~/.cache/transformers ]"
run_test "torch cache directory" "[ -d ~/.cache/torch ]"

echo ""
echo "=== Test 9: Workspace Setup ==="
run_test "abyss directory exists" "[ -d /workspaces/*/abyss ]"
run_test "abyss package importable" "cd /workspaces/*/abyss && python -c 'import abyss'" "true"

echo ""
echo "=== Test 10: Port Accessibility ==="
# Note: These tests check if ports are exposed, not necessarily if services are running
for port in 1883 9001 8883 9090; do
    run_test "Port $port exposed" "nc -z localhost $port 2>/dev/null" "true"
done

echo ""
echo "=== Test 11: Performance Settings ==="
# Check ulimits
NOFILE_LIMIT=$(ulimit -n)
run_test "File descriptor limit" "[ $NOFILE_LIMIT -ge 65536 ]" "true"

# Check shared memory
SHM_SIZE=$(df -h /dev/shm | tail -1 | awk '{print $2}')
echo "Shared memory size: $SHM_SIZE"

echo ""
echo "=== Test 12: Stress Test Readiness ==="
if [ -f /workspaces/*/abyss/src/abyss/run/uos_publish_wrapper.py ]; then
    run_test "Stress test script exists" "true"
    run_test "Stress test help accessible" "cd /workspaces/*/abyss && python src/abyss/run/uos_publish_wrapper.py --help | grep -q stress" "true"
else
    run_test "Stress test script exists" "false" "true"
fi

echo ""
echo "=== Test Summary ==="
echo -e "Tests Passed: ${GREEN}$PASSED${NC}"
echo -e "Tests Failed: ${RED}$FAILED${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All critical tests passed! ✓${NC}"
    echo "The devcontainer is ready for v0.2.6 development."
    exit 0
else
    echo -e "${RED}Some tests failed. Please check the configuration.${NC}"
    exit 1
fi