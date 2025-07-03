#!/bin/bash

# Publish Docker build script
# Builds publication/testing image for data publishing and testing workflows

set -e  # Exit on any error

# Configuration
IMAGE_NAME="uos-publish-json"
TAG="latest"
DOCKERFILE="Dockerfile.publish"
BUILD_CONTEXT="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting publish Docker build...${NC}"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}Error: $DOCKERFILE not found!${NC}"
    exit 1
fi

# Check if requirements file exists
if [ ! -f "abyss/requirements.txt" ]; then
    echo -e "${RED}Error: abyss/requirements.txt not found!${NC}"
    exit 1
fi

# Check if certificates exist
if [ ! -d "certs" ]; then
    echo -e "${YELLOW}Warning: certs/ directory not found. Creating placeholder...${NC}"
    mkdir -p certs
    touch certs/airbus-ca.pem
fi

# Display build information
echo -e "${GREEN}Build Configuration:${NC}"
echo "  Image: $IMAGE_NAME:$TAG"
echo "  Dockerfile: $DOCKERFILE"
echo "  Context: $BUILD_CONTEXT"
echo "  Base: python:3.10.16-slim"
echo "  Purpose: JSON data publishing and testing"
echo ""

# Clean up previous builds (optional)
echo -e "${YELLOW}Cleaning up previous builds...${NC}"
docker image prune -f --filter label=stage=intermediate 2>/dev/null || true

# Start build with timing
echo -e "${GREEN}Building Docker image...${NC}"
start_time=$(date +%s)

docker build \
    --file "$DOCKERFILE" \
    --tag "$IMAGE_NAME:$TAG" \
    --tag "$IMAGE_NAME:publisher" \
    --progress=plain \
    --no-cache \
    "$BUILD_CONTEXT"

end_time=$(date +%s)
build_duration=$((end_time - start_time))

echo -e "${GREEN}Build completed successfully!${NC}"
echo "Build duration: ${build_duration}s"

# Display image information
echo -e "${GREEN}Image Information:${NC}"
docker images "$IMAGE_NAME:$TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Optional: Test the built image
echo -e "${YELLOW}Testing built image...${NC}"
if docker run --rm "$IMAGE_NAME:$TAG" python -c "import abyss; print('Publisher image test passed!')"; then
    echo -e "${GREEN}Image test passed!${NC}"
else
    echo -e "${RED}Image test failed!${NC}"
    exit 1
fi

# Publisher-specific information
echo -e "${YELLOW}Publisher Features:${NC}"
echo "- Optimized for JSON data publishing workflows"
echo "- Includes uos_publish_json entry point"
echo "- Suitable for testing and data simulation"
echo "- Lightweight build for MQTT publishing tasks"
echo ""

# Display usage instructions
echo -e "${GREEN}Success! Usage instructions:${NC}"
echo ""
echo "Publish test data:"
echo "  docker run --rm -v \$(pwd)/test_data:/data $IMAGE_NAME:$TAG uos_publish_json /data"
echo ""
echo "Run interactively:"
echo "  docker run --rm -it $IMAGE_NAME:$TAG /bin/bash"
echo ""
echo "Run with network access to MQTT broker:"
echo "  docker run --rm --network toolbox-network $IMAGE_NAME:$TAG uos_publish_json /data"
echo ""
echo "Tag for registry:"
echo "  docker tag $IMAGE_NAME:$TAG your-registry/$IMAGE_NAME:$TAG"
echo ""
echo -e "${GREEN}Build completed successfully!${NC}"