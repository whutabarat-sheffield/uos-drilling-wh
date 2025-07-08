# GNU Make Build System - Quick Reference

## 🚀 **Most Common Commands**

```bash
# Get started
make help              # Show all available commands
make deps-check        # Check if everything is installed

# Build Docker images
make build-main        # Build main production image
make build-cpu         # Build CPU-only image (faster, smaller)
make build-all         # Build all images with testing

# Development workflow
make dev               # Complete development cycle
make dev-package       # Development cycle with wheel packaging
make dev-setup         # Set up development environment
make lint              # Check code quality
make format            # Auto-format code
make pytest            # Run tests

# Run containers
make run               # Run main container
make run-devel         # Run development container

# Cleanup
make clean-all         # Clean everything
make clean-wheel       # Clean wheel artifacts
```

## ⚡ **Performance Optimized Commands**

```bash
# Fast parallel builds (4x faster)
make build-all-parallel

# Super fast builds (skip testing)
make build-all-quick

# Quick development check
make dev-test          # Just lint + test (no build)
```

## 🔧 **Advanced Usage**

```bash
# Registry operations
make push REGISTRY=your-registry.com
make pull REGISTRY=your-registry.com

# Compose operations
make compose-up        # Start full dev environment
make compose-down      # Stop dev environment

# Python wheel packaging
make build-wheel       # Build Python wheel distribution
make install-wheel     # Install wheel from local build

# Information and debugging
make info              # Project information
make version           # Version details
make benchmark         # Performance stats
make logs              # Container logs
make shell             # Open container shell
```

## 🆚 **Make vs Shell Scripts**

| Task | Make Command | Legacy Shell Script | Advantage |
|------|--------------|-------------------|-----------|
| Build main | `make build-main` | `./build-main.sh` | ✅ Testing included |
| Build all | `make build-all-parallel` | `./build-all.sh` | ✅ 4x faster |
| Development | `make dev` | Multiple scripts | ✅ Integrated workflow |
| Cleanup | `make clean-all` | Manual commands | ✅ Complete cleanup |

## 📚 **Getting Help**

- `make help` - Complete help system
- `make deps-check` - Check system requirements  
- `docs/BUILD-GUIDE.md` - Detailed build documentation
- `CLAUDE.md` - Complete project documentation

## 🔄 **Migration from Shell Scripts**

**Old way:**
```bash
./build-main.sh
./build-cpu.sh
./build-all.sh
```

**New way:**
```bash
make build-main
make build-cpu
make build-all-parallel  # Much faster!
```

**Both work!** Shell scripts are maintained for backward compatibility.

## 🎯 **Quick Start for New Developers**

```bash
# 1. Check environment
make deps-check

# 2. Set up development
make dev-setup

# 3. Build and test everything
make dev

# 4. Start working
make run-devel
```

## ⚠️ **Troubleshooting**

- `make deps-check` - Verify environment
- `make dev-check` - Check development setup
- `make clean-all` - Reset everything
- Legacy scripts still work: `./build-main.sh`

For detailed troubleshooting, see `CLAUDE.md` → Troubleshooting Guide.