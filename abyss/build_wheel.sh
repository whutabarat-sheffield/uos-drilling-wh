#!/bin/bash
# Build script wrapper for creating Python wheels

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default options
CLEAN=false
INSTALL=false
EDITABLE=false
SKIP_VALIDATION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --clean)
      CLEAN=true
      shift
      ;;
    --install)
      INSTALL=true
      shift
      ;;
    --editable)
      EDITABLE=true
      INSTALL=true  # Editable requires install
      shift
      ;;
    --skip-validation)
      SKIP_VALIDATION=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --clean             Clean build artifacts before building"
      echo "  --install           Install the built wheel after building"
      echo "  --editable          Install in editable mode (implies --install)"
      echo "  --skip-validation   Skip wheel validation"
      echo "  --help, -h          Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                    # Build wheel only"
      echo "  $0 --clean           # Clean and build"
      echo "  $0 --clean --install # Clean, build, and install"
      echo "  $0 --editable        # Build and install in editable mode"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Build the arguments for the Python script
PYTHON_ARGS=()

if [ "$CLEAN" = true ]; then
    PYTHON_ARGS+=(--clean)
fi

if [ "$INSTALL" = true ]; then
    PYTHON_ARGS+=(--install)
fi

if [ "$EDITABLE" = true ]; then
    PYTHON_ARGS+=(--editable)
fi

if [ "$SKIP_VALIDATION" = true ]; then
    PYTHON_ARGS+=(--skip-validation)
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Run the Python build script
echo "Starting wheel build process..."
echo "Arguments: ${PYTHON_ARGS[*]}"
echo ""

python3 build_wheel.py "${PYTHON_ARGS[@]}"

# Show final status
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Wheel build completed successfully!"
    echo ""
    echo "Built wheels are in the 'dist/' directory:"
    ls -la dist/*.whl 2>/dev/null || echo "No wheels found in dist/"
else
    echo ""
    echo "❌ Wheel build failed!"
    exit 1
fi