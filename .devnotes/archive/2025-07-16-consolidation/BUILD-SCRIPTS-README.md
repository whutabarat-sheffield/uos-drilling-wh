# Docker Build Scripts

Comprehensive build scripts for all Docker configurations in the UOS drilling depth estimation system.

## Available Build Scripts

### 🔧 Individual Build Scripts

| Script | Dockerfile | Purpose | Base Image | Features |
|--------|------------|---------|------------|----------|
| `build-main.sh` | `Dockerfile` | Standard production build | `python:3.10.16-slim` | PyTorch CPU, standard config |
| `build-cpu.sh` | `Dockerfile.cpu` | CPU-optimized build | `python:3.10.16-slim` | No CUDA, optimized dependencies |
| `build-runtime.sh` | `Dockerfile.runtime` | GPU-enabled runtime | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` | Full GPU support |
| `build-devel.sh` | `Dockerfile.devel` | Development environment | `python:3.10.16-slim` | Debug tools, interactive |
| `build-publish.sh` | `Dockerfile.publish` | Data publishing | `python:3.10.16-slim` | JSON publishing, testing |

### 🚀 Batch Build Script

| Script | Purpose | Features |
|--------|---------|----------|
| `build-all.sh` | Build all configurations | Progress tracking, error handling, summary report |

## Quick Start

### Build Individual Images
```bash
# CPU-optimized (recommended for production)
./build-cpu.sh

# Standard build
./build-main.sh

# GPU-enabled runtime
./build-runtime.sh

# Development environment
./build-devel.sh

# Data publisher
./build-publish.sh
```

### Build All Images
```bash
# Build everything at once
./build-all.sh
```

## Script Features

### 🔍 Validation & Error Handling
- ✅ Dockerfile existence check
- ✅ Requirements file validation
- ✅ Certificate handling (with fallbacks)
- ✅ Build error detection and reporting
- ✅ Image functionality testing

### 📊 Build Information
- ✅ Build timing and duration tracking
- ✅ Image size reporting
- ✅ Build configuration display
- ✅ Success/failure status
- ✅ Usage instructions

### 🧪 Automatic Testing
- ✅ Python import verification
- ✅ PyTorch installation check
- ✅ CUDA availability detection
- ✅ Package functionality validation

## Build Outputs

### Expected Image Sizes
| Build Type | Expected Size | Use Case |
|------------|---------------|----------|
| **CPU-optimized** | ~3.5GB | Production CPU-only deployment |
| **Standard** | ~10GB | Standard production with PyTorch |
| **Runtime** | ~13GB | GPU-enabled production |
| **Development** | ~10GB | Interactive development |
| **Publisher** | ~3GB | Data publishing and testing |

### Generated Image Tags
Each build creates multiple tags for flexibility:

```bash
# CPU build
uos-depthest-listener:cpu
uos-depthest-listener:latest-cpu

# Main build  
uos-depthest-listener:latest
uos-depthest-listener:main

# Runtime build
uos-depthest-listener:runtime
uos-depthest-listener:gpu

# Development build
uos-depthest-listener:devel
uos-depthest-listener:dev

# Publisher build
uos-publish-json:latest
uos-publish-json:publisher
```

## Usage Examples

### Individual Build Usage
```bash
# Build and test CPU-optimized image
./build-cpu.sh

# Run the built image
docker run --rm -it uos-depthest-listener:cpu

# Run with Docker Compose
docker-compose -f docker-compose.cpu.yml up
```

### Batch Build Usage
```bash
# Build all images and get summary
./build-all.sh

# Check results
docker images | grep uos
```

### Development Workflow
```bash
# Build development image
./build-devel.sh

# Run interactively for debugging
docker run --rm -it -v $(pwd)/abyss:/app/abyss uos-depthest-listener:devel /bin/bash
```

### Testing Workflow
```bash
# Build publisher for testing
./build-publish.sh

# Publish test data
docker run --rm -v $(pwd)/test_data:/data uos-publish-json:latest uos_publish_json /data
```

## Build Script Options

### Environment Variables
```bash
# Customize image name and tag
IMAGE_NAME="custom-listener" TAG="v1.0" ./build-cpu.sh

# Skip image pruning
SKIP_CLEANUP=1 ./build-main.sh

# Verbose output
VERBOSE=1 ./build-all.sh
```

### Build Arguments
Scripts support Docker build arguments:
```bash
# Custom Python version
PYTHON_VERSION=3.11 ./build-main.sh

# Custom UID for non-root user
UID=1000 ./build-devel.sh
```

## Troubleshooting

### Common Issues

1. **Certificate Errors**
   ```bash
   # Create certificates directory
   mkdir -p certs
   touch certs/airbus-ca.pem
   ```

2. **Permission Denied**
   ```bash
   # Make scripts executable
   chmod +x build-*.sh
   ```

3. **Disk Space**
   ```bash
   # Clean up before building
   docker system prune -a
   ```

4. **Build Cache Issues**
   ```bash
   # Force rebuild without cache
   docker build --no-cache ...
   ```

### Build Validation
Each script automatically validates:
- ✅ Image builds successfully
- ✅ Python imports work
- ✅ PyTorch loads correctly
- ✅ CUDA detection (where applicable)
- ✅ Application entry points

### Log Analysis
Build logs include:
- Timing information
- Size comparisons
- Dependency resolution
- Error details with context
- Success/failure summary

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Build Docker Images
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build all images
      run: ./build-all.sh
    - name: Test images
      run: docker run --rm uos-depthest-listener:cpu python -c "import abyss"
```

### Production Deployment
```bash
# Build production image
./build-cpu.sh

# Tag for registry
docker tag uos-depthest-listener:cpu registry.company.com/uos-depthest-listener:latest

# Push to registry
docker push registry.company.com/uos-depthest-listener:latest
```

## Maintenance

### Script Updates
To update build scripts:
1. Modify individual scripts as needed
2. Test with `./build-all.sh`
3. Update this documentation
4. Commit changes

### Adding New Configurations
1. Create new `Dockerfile.newconfig`
2. Create corresponding `build-newconfig.sh` 
3. Add to `build-all.sh` script list
4. Update documentation

These build scripts provide a complete, automated solution for managing all Docker configurations in the UOS drilling depth estimation system.