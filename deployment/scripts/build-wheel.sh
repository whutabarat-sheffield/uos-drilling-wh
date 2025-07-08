#!/bin/bash

# Python wheel build script
# Builds Python wheel distribution for the abyss package

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}    Building Python Wheel Distribution    ${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Configuration
PACKAGE_DIR="abyss"
DIST_DIR="$PACKAGE_DIR/dist"
BUILD_DIR="$PACKAGE_DIR/build"
EGG_INFO_DIR="$PACKAGE_DIR/src/abyss.egg-info"

# Check if we're in the correct directory
if [ ! -f "$PACKAGE_DIR/pyproject.toml" ]; then
    echo -e "${RED}Error: $PACKAGE_DIR/pyproject.toml not found!${NC}"
    echo -e "${YELLOW}Please run this script from the project root directory${NC}"
    exit 1
fi

# Check if requirements file exists
if [ ! -f "$PACKAGE_DIR/requirements.txt" ]; then
    echo -e "${RED}Error: $PACKAGE_DIR/requirements.txt not found!${NC}"
    exit 1
fi

# Display package information
echo -e "${GREEN}Package Information:${NC}"
cd "$PACKAGE_DIR"
python -c "
try:
    import tomllib
except ImportError:
    import tomli as tomllib
    
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
    project = data['project']
    print(f'  Name: {project[\"name\"]}')
    print(f'  Version: {project[\"version\"]}')
    print(f'  Description: {project[\"description\"]}')
    print(f'  Python: {project[\"requires-python\"]}')
"
cd ..
echo ""

# Clean up previous builds
echo -e "${YELLOW}Cleaning up previous builds...${NC}"
rm -rf "$DIST_DIR" "$BUILD_DIR" "$EGG_INFO_DIR"
echo "  - Removed: $DIST_DIR"
echo "  - Removed: $BUILD_DIR"
echo "  - Removed: $EGG_INFO_DIR"
echo ""

# Check and install build dependencies
echo -e "${YELLOW}Checking build dependencies...${NC}"
cd "$PACKAGE_DIR"

# Install build dependencies if needed
python -m pip install --upgrade pip setuptools wheel build 2>/dev/null || {
    echo -e "${RED}Error: Failed to install build dependencies${NC}"
    exit 1
}

echo -e "${GREEN}✅ Build dependencies available${NC}"
echo ""

# Build wheel
echo -e "${GREEN}Building wheel distribution...${NC}"
start_time=$(date +%s)

# Use python -m build for modern wheel building
python -m build --wheel --outdir dist/ . || {
    echo -e "${RED}Error: Wheel build failed${NC}"
    exit 1
}

end_time=$(date +%s)
build_duration=$((end_time - start_time))

echo -e "${GREEN}✅ Wheel build completed successfully!${NC}"
echo "Build duration: ${build_duration}s"
echo ""

# Display build results
echo -e "${GREEN}Build Results:${NC}"
if [ -d "dist" ]; then
    echo "Distribution files created in: $(pwd)/dist/"
    ls -la dist/
    echo ""
    
    # Show wheel information
    wheel_file=$(find dist/ -name "*.whl" | head -1)
    if [ -n "$wheel_file" ]; then
        echo -e "${BLUE}Wheel Information:${NC}"
        file_size=$(du -h "$wheel_file" | cut -f1)
        echo "  File: $(basename "$wheel_file")"
        echo "  Size: $file_size"
        echo "  Path: $(realpath "$wheel_file")"
        echo ""
        
        # Verify wheel contents
        echo -e "${BLUE}Wheel Contents:${NC}"
        python -m zipfile -l "$wheel_file" | head -20
        echo ""
    fi
else
    echo -e "${RED}Error: No distribution files found${NC}"
    exit 1
fi

# Test wheel installation (optional)
echo -e "${YELLOW}Testing wheel installation...${NC}"
pip install --force-reinstall --no-deps "$wheel_file" && {
    echo -e "${GREEN}✅ Wheel installation test passed${NC}"
} || {
    echo -e "${YELLOW}⚠️  Wheel installation test failed${NC}"
}
echo ""

# Back to project root
cd ..

# Display usage instructions
echo -e "${GREEN}Success! Usage instructions:${NC}"
echo ""
echo "Install the wheel:"
echo "  pip install $PACKAGE_DIR/dist/abyss-*.whl"
echo ""
echo "Upload to PyPI (if configured):"
echo "  python -m twine upload $PACKAGE_DIR/dist/abyss-*.whl"
echo ""
echo "Development installation:"
echo "  cd $PACKAGE_DIR && pip install -e ."
echo ""
echo -e "${GREEN}Wheel build completed successfully!${NC}"