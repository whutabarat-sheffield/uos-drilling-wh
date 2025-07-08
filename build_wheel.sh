#!/bin/bash
#
# Build Wheel Script for UOS Drilling Depth Estimation System
#
# This script provides a simple wrapper around the Python wheel builder.
# It builds the abyss package into a wheel distribution using pyproject.toml.
#
# Usage:
#   ./build_wheel.sh [options]
#
# Options:
#   --clean         Clean build artifacts before building
#   --validate      Validate the built wheel
#   --output-dir    Specify output directory for wheels (default: dist/)
#   --verbose       Enable verbose output
#   --help          Show help message
#
# Examples:
#   ./build_wheel.sh
#   ./build_wheel.sh --clean --validate
#   ./build_wheel.sh --output-dir=wheels/ --verbose

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BUILDER="$SCRIPT_DIR/build_wheel.py"

# Check if Python builder exists
if [[ ! -f "$PYTHON_BUILDER" ]]; then
    print_error "Python builder script not found: $PYTHON_BUILDER"
    exit 1
fi

# Check Python availability
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not found in PATH"
    exit 1
fi

# Print header
echo "=============================================================="
echo "UOS Drilling Depth Estimation System - Wheel Builder"
echo "=============================================================="
echo

print_status "Using Python: $(python3 --version)"
print_status "Build script: $PYTHON_BUILDER"
echo

# Pass all arguments to the Python builder
print_status "Starting wheel build process..."
if python3 "$PYTHON_BUILDER" "$@"; then
    echo
    print_success "Wheel build completed successfully!"
    
    # Show built wheels if default output directory is used
    if [[ -d "$SCRIPT_DIR/dist" ]] && [[ ! "$*" =~ --output-dir ]]; then
        echo
        print_status "Built wheels in dist/ directory:"
        ls -la "$SCRIPT_DIR/dist/"*.whl 2>/dev/null || print_warning "No wheel files found in dist/"
    fi
else
    echo
    print_error "Wheel build failed!"
    exit 1
fi

echo
echo "=============================================================="
echo "Build process completed"
echo "=============================================================="