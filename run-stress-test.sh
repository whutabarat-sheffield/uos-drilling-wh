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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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
    -n, --no-sleep              Disable rate limiting for maximum throughput
    -b, --build                 Build Docker image before running
    -l, --local                 Run locally instead of in Docker
    -h, --help                  Show this help message

EXAMPLES:
    # Run default stress test (1000 signals/sec for 60 seconds)
    $0

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
NO_SLEEP=""
BUILD=false
LOCAL=false

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
print_info "  Mode: $([ "$LOCAL" = true ] && echo "Local" || echo "Docker")"
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
    
    CMD="python uos_publish_wrapper.py ../$DATA_PATH \
        --stress-test \
        --rate $RATE \
        --duration $DURATION \
        --concurrent-publishers $PUBLISHERS \
        -c ../$CONFIG \
        --log-level INFO \
        $NO_SLEEP"
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
    if ! docker image inspect uos-publish:latest >/dev/null 2>&1; then
        print_error "Docker image 'uos-publish:latest' not found. Run with -b to build."
        exit 1
    fi
    
    CMD="docker run --rm --name uos-stress-test \
        --network mqtt-multistack_default \
        uos-publish:latest \
        python uos_publish_wrapper.py test_data \
        --stress-test \
        --rate $RATE \
        --duration $DURATION \
        --concurrent-publishers $PUBLISHERS \
        --log-level INFO \
        $NO_SLEEP"
fi

# Run the stress test
print_info "Starting stress test..."
echo
eval $CMD

# Check exit code
if [ $? -eq 0 ]; then
    print_info "Stress test completed successfully"
else
    print_error "Stress test failed"
    exit 1
fi