#!/bin/bash
# Build script for lightweight MQTT publisher container

set -e

echo "Building lightweight MQTT publisher container..."
echo "============================================="

# Build the lightweight publisher image
docker build -f Dockerfile.publisher -t abyss-publisher:lightweight .

echo ""
echo "Build complete! Image details:"
docker images abyss-publisher:lightweight

echo ""
echo "To run the publisher:"
echo "  docker run -it abyss-publisher:lightweight"
echo ""
echo "To run with custom data:"
echo "  docker run -it -v /path/to/your/data:/data abyss-publisher:lightweight /data"
echo ""
echo "To run with custom config:"
echo "  docker run -it -v /path/to/config.yaml:/app/config/custom.yaml abyss-publisher:lightweight test_data -c /app/config/custom.yaml"
echo ""
echo "Image size comparison:"
echo "  Full system:  ~2GB"
echo "  Lightweight:  ~100MB (95% reduction!)"