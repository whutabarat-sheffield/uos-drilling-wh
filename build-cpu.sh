#!/bin/bash

# CPU-only Docker build script
# Builds optimized Docker image without CUDA dependencies

set -e  # Exit on any error

# Configuration
IMAGE_NAME="uos-depthest-listener"
TAG="cpu"
DOCKERFILE="Dockerfile.cpu"
BUILD_CONTEXT="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting CPU-only Docker build...${NC}"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}Error: $DOCKERFILE not found!${NC}"
    exit 1
fi

# Check if requirements file exists
if [ ! -f "abyss/requirements.cpu" ]; then
    echo -e "${RED}Error: abyss/requirements.cpu not found!${NC}"
    exit 1
fi

# Note: This CPU-only build does not require certificates
# The Dockerfile uses --trusted-host flags to bypass certificate validation

# Display build information
echo -e "${GREEN}Build Configuration:${NC}"
echo "  Image: $IMAGE_NAME:$TAG"
echo "  Dockerfile: $DOCKERFILE"
echo "  Context: $BUILD_CONTEXT"
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
    --tag "$IMAGE_NAME:latest-cpu" \
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
if docker run --rm "$IMAGE_NAME:$TAG" python -c "import abyss; import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print('Image test passed!')"; then
    echo -e "${GREEN}Image test passed!${NC}"
else
    echo -e "${RED}Image test failed!${NC}"
    exit 1
fi

# Display usage instructions
echo -e "${GREEN}Success! Usage instructions:${NC}"
echo ""
echo "Run the container:"
echo "  docker run --rm -it $IMAGE_NAME:$TAG"
echo ""
echo "Run with Docker Compose:"
echo "  docker-compose -f docker-compose.cpu.yml up"
echo ""
echo "Tag for registry:"
echo "  docker tag $IMAGE_NAME:$TAG your-registry/$IMAGE_NAME:$TAG"
echo ""
echo -e "${GREEN}Build completed successfully!${NC}"