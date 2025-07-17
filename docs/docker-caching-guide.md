# Docker Build Caching Guide

## Overview

The Docker build scripts have been optimized to use Docker's layer caching mechanism, which significantly reduces build times and bandwidth usage by reusing previously built layers when files haven't changed.

## Key Improvements

### 1. **BuildKit Enabled**
All build scripts now set `export DOCKER_BUILDKIT=1` to enable Docker BuildKit, which provides:
- Better build performance
- Advanced caching features
- Mount-based caching for package managers

### 2. **Cache-Enabled by Default**
- The `--no-cache` flag has been removed from default builds
- Builds now use Docker's layer cache automatically
- Subsequent builds only rebuild layers that have changed

### 3. **Optional Fresh Builds**
All build scripts now accept a `--no-cache` flag when you need a fresh build:
```bash
./build-main.sh --no-cache    # Force a fresh build without cache
./build-all.sh --no-cache     # Fresh build for all configurations
```

## Performance Impact

With caching enabled, typical build time improvements:
- **First build**: ~10-15 minutes (downloads all dependencies)
- **Subsequent builds** (no changes): ~30 seconds
- **Builds with code changes**: ~1-2 minutes (only rebuilds affected layers)

## Cache Management

### Using the Cache Cleanup Script
```bash
./clean-cache.sh              # Interactive cache cleanup
./clean-cache.sh --force      # Skip confirmation prompts
./clean-cache.sh --all        # Remove all images (use with caution)
```

### Manual Cache Management
```bash
# View Docker disk usage
docker system df

# Clean build cache only
docker builder prune

# Clean all unused data
docker system prune -a
```

## Best Practices

1. **Regular Builds**: Just run the build script normally to benefit from caching
   ```bash
   ./build-main.sh
   ```

2. **Fresh Builds**: Use `--no-cache` when you need to ensure all dependencies are freshly downloaded
   ```bash
   ./build-main.sh --no-cache
   ```

3. **Cache Maintenance**: Run the cleanup script periodically to manage disk space
   ```bash
   ./clean-cache.sh
   ```

4. **CI/CD Considerations**: 
   - Consider using `--cache-from` and `--cache-to` for distributed caching
   - Use registry-based caching for shared build environments

## How Docker Layer Caching Works

Docker creates a cache layer for each instruction in the Dockerfile:
1. If the instruction and its context haven't changed, Docker reuses the cached layer
2. Once a layer changes, all subsequent layers must be rebuilt
3. The build context (files being copied) affects cache validity

## Troubleshooting

### Cache Not Working?
- Ensure BuildKit is enabled: `echo $DOCKER_BUILDKIT` should show `1`
- Check if files are being modified unnecessarily (timestamps, etc.)
- Use `docker build --progress=plain` to see detailed cache hit/miss info

### Disk Space Issues?
- Run `./clean-cache.sh` to clean up build cache
- Use `docker system df` to analyze disk usage
- Consider `docker system prune -a` for aggressive cleanup

### Need Reproducible Builds?
- Use `--no-cache` flag to ensure fresh downloads
- Consider using specific version tags for base images
- Pin dependency versions in requirements files