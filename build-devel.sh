#!/bin/bash

# Development Docker build script
# Builds development image with additional tools and debugging capabilities

set -e  # Exit on any error

# Configuration
IMAGE_NAME="uos-depthest-listener"
TAG="devel"
DOCKERFILE="Dockerfile.devel"
BUILD_CONTEXT="."

# Enable BuildKit for better caching
export DOCKER_BUILDKIT=1

# Parse command line arguments
NO_CACHE=""
for arg in "$@"; do
    case $arg in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --no-cache    Force rebuild without using cache"
            echo "  --help, -h    Show this help message"
            exit 0
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting development Docker build...${NC}"
if [ -n "$NO_CACHE" ]; then
    echo -e "${YELLOW}Building without cache (--no-cache flag set)${NC}"
else
    echo -e "${GREEN}Using Docker build cache for faster builds${NC}"
fi

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
echo "  Purpose: Development and debugging"
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
    --tag "$IMAGE_NAME:dev" \
    --progress=plain \
    $NO_CACHE \
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

# Development-specific information
echo -e "${YELLOW}Development Features:${NC}"
echo "- Non-privileged user setup for security"
echo "- Additional development tools available"
echo "- Suitable for interactive development and debugging"
echo ""

# Display usage instructions
echo -e "${GREEN}Success! Usage instructions:${NC}"
echo ""
echo "Run interactively for development:"
echo "  docker run --rm -it $IMAGE_NAME:$TAG /bin/bash"
echo ""
echo "Run with volume mounts for development:"
echo "  docker run --rm -it -v \$(pwd)/abyss:/app/abyss $IMAGE_NAME:$TAG"
echo ""
echo "Run the application:"
echo "  docker run --rm -it $IMAGE_NAME:$TAG"
echo ""
echo "Tag for registry:"
echo "  docker tag $IMAGE_NAME:$TAG your-registry/$IMAGE_NAME:$TAG"
echo ""
echo -e "${GREEN}Build completed successfully!${NC}"