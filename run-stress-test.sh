#!/bin/bash
# Stress Test Runner for UOS MQTT Publisher
# This script helps run stress tests with various configurations

set -e

# Default values
DEFAULT_RATE=1000
DEFAULT_DURATION=60
DEFAULT_PUBLISHERS=10
DEFAULT_DATA_PATH="abyss/src/abyss/test_data"
DEFAULT_CONFIG="abyss/src/abyss/run/config/mqtt_conf_docker.yaml"
DEFAULT_NETWORK="auto"  # Changed to auto-detect by default

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Function to detect MQTT-related Docker networks
detect_mqtt_network() {
    print_debug "Detecting MQTT broker network..."
    
    # Try common network patterns in order of preference
    local networks=(
        "mqtt-broker_toolbox-network"  # Portainer multi-stack
        "toolbox-network"              # Direct docker-compose
        "mqtt_default"                 # Common MQTT stack name
        "mqtt-network"                 # Alternative naming
    )
    
    for net in "${networks[@]}"; do
        if docker network inspect "$net" >/dev/null 2>&1; then
            print_info "Found network: $net"
            echo "$net"
            return 0
        fi
    done
    
    # Check if MQTT broker is running on any network
    print_debug "Checking for MQTT broker container..."
    local broker_network=$(docker inspect mqtt-broker 2>/dev/null | grep -o '"NetworkMode": "[^"]*"' | cut -d'"' -f4)
    if [ -n "$broker_network" ] && [ "$broker_network" != "null" ]; then
        print_info "Found MQTT broker on network: $broker_network"
        echo "$broker_network"
        return 0
    fi
    
    # No suitable network found
    print_warn "No MQTT-related network found"
    return 1
}

# Function to test MQTT broker connectivity
test_mqtt_connectivity() {
    local network=$1
    local test_container="mqtt-connectivity-test-$$"
    
    print_debug "Testing MQTT broker connectivity on network: $network"
    
    # Run a quick test using the publisher image
    if docker run --rm --name "$test_container" \
        --network "$network" \
        uos-publish-json:latest \
        python -c "
import paho.mqtt.client as mqtt
try:
    # Try new API (paho-mqtt >= 2.0)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, 'test')
except (AttributeError, TypeError):
    # Fall back to old API (paho-mqtt < 2.0)
    client = mqtt.Client('test')
try:
    client.connect('mqtt-broker', 1883, 60)
    print('SUCCESS')
except:
    exit(1)
" 2>/dev/null | grep -q "SUCCESS"; then
        return 0
    else
        return 1
    fi
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run MQTT publisher stress tests with various configurations.

OPTIONS:
    -r, --rate RATE              Target signals per second (default: $DEFAULT_RATE)
    -d, --duration SECONDS       Test duration in seconds (default: $DEFAULT_DURATION)
    -p, --publishers COUNT       Number of concurrent publishers (default: $DEFAULT_PUBLISHERS)
    -D, --data PATH             Path to test data (default: $DEFAULT_DATA_PATH)
    -c, --config FILE           Configuration file (default: $DEFAULT_CONFIG)
    -N, --network NAME          Docker network name (default: auto-detect)
    -n, --no-sleep              Disable rate limiting for maximum throughput
    -b, --build                 Build Docker image before running
    -l, --local                 Run locally instead of in Docker
    --standalone                Run with host network (no Docker network)
    --create-network            Create a temporary network for testing
    --async                    Use async mode for ultra-high performance
    --auto-detect              Auto-detect MQTT network (default)
    -h, --help                  Show this help message

EXAMPLES:
    # Run default stress test (1000 signals/sec for 60 seconds)
    $0

    # Run async mode for maximum performance
    $0 --async -r 1000 -p 50

    # Run high-load test with 20 publishers
    $0 -r 1000 -p 20 -n

    # Run 5-minute endurance test
    $0 -r 500 -d 300

    # Build and run in Docker
    $0 -b -r 1000

EOF
    exit 1
}

# Parse command line arguments
RATE=$DEFAULT_RATE
DURATION=$DEFAULT_DURATION
PUBLISHERS=$DEFAULT_PUBLISHERS
DATA_PATH=$DEFAULT_DATA_PATH
CONFIG=$DEFAULT_CONFIG
NETWORK=$DEFAULT_NETWORK
NO_SLEEP=""
BUILD=false
LOCAL=false
STANDALONE=false
CREATE_NETWORK=false
AUTO_DETECT=true
ASYNC_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rate)
            RATE="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -p|--publishers)
            PUBLISHERS="$2"
            shift 2
            ;;
        -D|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -N|--network)
            NETWORK="$2"
            AUTO_DETECT=false
            shift 2
            ;;
        -n|--no-sleep)
            NO_SLEEP="--no-sleep"
            shift
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -l|--local)
            LOCAL=true
            shift
            ;;
        --standalone)
            STANDALONE=true
            AUTO_DETECT=false
            shift
            ;;
        --create-network)
            CREATE_NETWORK=true
            AUTO_DETECT=false
            shift
            ;;
        --async)
            ASYNC_MODE=true
            shift
            ;;
        --auto-detect)
            AUTO_DETECT=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate inputs
if ! [[ "$RATE" =~ ^[0-9]+$ ]] || [ "$RATE" -lt 1 ]; then
    print_error "Rate must be a positive integer"
    exit 1
fi

if ! [[ "$DURATION" =~ ^[0-9]+$ ]] || [ "$DURATION" -lt 1 ]; then
    print_error "Duration must be a positive integer"
    exit 1
fi

if ! [[ "$PUBLISHERS" =~ ^[0-9]+$ ]] || [ "$PUBLISHERS" -lt 1 ]; then
    print_error "Publishers count must be a positive integer"
    exit 1
fi

# Print test configuration
print_info "Stress Test Configuration:"
print_info "  Target Rate: $RATE signals/second"
print_info "  Duration: $DURATION seconds"
print_info "  Publishers: $PUBLISHERS concurrent threads"
print_info "  Data Path: $DATA_PATH"
print_info "  Config: $CONFIG"
print_info "  No Sleep: $([ -n "$NO_SLEEP" ] && echo "Yes" || echo "No")"
print_info "  Async Mode: $([ "$ASYNC_MODE" = true ] && echo "Yes" || echo "No")"
print_info "  Mode: $([ "$LOCAL" = true ] && echo "Local" || echo "Docker")"
if [ "$LOCAL" = false ]; then
    if [ "$STANDALONE" = true ]; then
        print_info "  Network Mode: Standalone (host network)"
    elif [ "$CREATE_NETWORK" = true ]; then
        print_info "  Network Mode: Create temporary network"
    elif [ "$AUTO_DETECT" = true ]; then
        print_info "  Network Mode: Auto-detect"
    else
        print_info "  Network: $NETWORK"
    fi
fi
echo

# Build command
if [ "$LOCAL" = true ]; then
    # Run locally
    if [ ! -f "$CONFIG" ]; then
        print_error "Configuration file not found: $CONFIG"
        exit 1
    fi
    
    if [ ! -d "$DATA_PATH" ]; then
        print_error "Data path not found: $DATA_PATH"
        exit 1
    fi
    
    # Change to the run directory
    cd abyss/src/abyss/run
    
    # Build command based on mode
    if [ "$ASYNC_MODE" = true ]; then
        CMD="python uos_publish_wrapper.py ../$DATA_PATH \
            --async \
            --rate $RATE \
            --duration $DURATION \
            --concurrent-publishers $PUBLISHERS \
            -c ../$CONFIG \
            --log-level INFO \
            $NO_SLEEP"
    else
        CMD="python uos_publish_wrapper.py ../$DATA_PATH \
            --stress-test \
            --rate $RATE \
            --duration $DURATION \
            --concurrent-publishers $PUBLISHERS \
            -c ../$CONFIG \
            --log-level INFO \
            $NO_SLEEP"
    fi
else
    # Run in Docker
    if [ "$BUILD" = true ]; then
        print_info "Building Docker image..."
        ./build-publish.sh
        if [ $? -ne 0 ]; then
            print_error "Docker build failed"
            exit 1
        fi
    fi
    
    # Check if image exists
    if ! docker image inspect uos-publish-json:latest >/dev/null 2>&1; then
        print_error "Docker image 'uos-publish-json:latest' not found. Run with -b to build."
        exit 1
    fi
    
    # Handle network configuration
    NETWORK_ARG=""
    TEMP_NETWORK=""
    
    if [ "$STANDALONE" = true ]; then
        # Use host network
        NETWORK_ARG="--network host"
        print_info "Using host network mode"
        # Update config to use localhost when in host mode
        CONFIG="abyss/src/abyss/run/config/mqtt_conf_local.yaml"
        print_info "Switched to local config for host network mode"
    elif [ "$CREATE_NETWORK" = true ]; then
        # Create temporary network
        TEMP_NETWORK="stress-test-network-$$"
        print_info "Creating temporary network: $TEMP_NETWORK"
        docker network create "$TEMP_NETWORK" >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            print_error "Failed to create temporary network"
            exit 1
        fi
        NETWORK_ARG="--network $TEMP_NETWORK"
        print_warn "Temporary network created, but MQTT broker may not be accessible"
    elif [ "$AUTO_DETECT" = true ] && [ "$NETWORK" = "auto" ]; then
        # Auto-detect network
        DETECTED_NETWORK=$(detect_mqtt_network)
        if [ $? -eq 0 ]; then
            NETWORK="$DETECTED_NETWORK"
            NETWORK_ARG="--network $NETWORK"
            
            # Test connectivity
            if ! test_mqtt_connectivity "$NETWORK"; then
                print_warn "MQTT broker not accessible on network $NETWORK"
                print_info "Falling back to host network mode"
                NETWORK_ARG="--network host"
                CONFIG="abyss/src/abyss/run/config/mqtt_conf_local.yaml"
            fi
        else
            print_warn "No MQTT network detected, using host network"
            NETWORK_ARG="--network host"
            CONFIG="abyss/src/abyss/run/config/mqtt_conf_local.yaml"
        fi
    else
        # Use specified network
        NETWORK_ARG="--network $NETWORK"
        # Verify network exists
        if ! docker network inspect "$NETWORK" >/dev/null 2>&1; then
            print_error "Network '$NETWORK' does not exist"
            print_info "Available networks:"
            docker network ls --format "  - {{.Name}}"
            exit 1
        fi
    fi
    
    # Build Docker command based on mode
    if [ "$ASYNC_MODE" = true ]; then
        MODE_FLAG="--async"
    else
        MODE_FLAG="--stress-test"
    fi
    
    CMD="docker run --rm --name uos-stress-test \
        $NETWORK_ARG \
        uos-publish-json:latest \
        python uos_publish_wrapper.py test_data \
        $MODE_FLAG \
        --rate $RATE \
        --duration $DURATION \
        --concurrent-publishers $PUBLISHERS \
        -c /app/config/$(basename $CONFIG) \
        --log-level INFO \
        $NO_SLEEP"
fi

# Run the stress test
print_info "Starting stress test..."
echo
eval $CMD

# Check exit code
EXIT_CODE=$?

# Cleanup temporary network if created
if [ -n "$TEMP_NETWORK" ]; then
    print_info "Cleaning up temporary network: $TEMP_NETWORK"
    docker network rm "$TEMP_NETWORK" >/dev/null 2>&1
fi

if [ $EXIT_CODE -eq 0 ]; then
    print_info "Stress test completed successfully"
else
    print_error "Stress test failed"
    exit 1
fi