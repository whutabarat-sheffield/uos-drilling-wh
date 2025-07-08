# Building abyss Package

This document describes how to build Python wheels for the abyss package.

## Quick Start

### Using the Build Script (Recommended)

```bash
# Build wheel only
./build_wheel.sh

# Clean build artifacts and build
./build_wheel.sh --clean

# Build and install in development mode
./build_wheel.sh --editable

# Build and install normally
./build_wheel.sh --clean --install
```

### Using Make

```bash
# Build wheel
make build

# Clean and build
make clean build

# Install in development mode
make dev-install

# Run tests
make test

# Show all available targets
make help
```

### Using Python Directly

```bash
# Build wheel
python3 build_wheel.py

# Clean and build with installation
python3 build_wheel.py --clean --install --editable
```

## Build Requirements

The build process requires the following Python packages:

- `build` - Modern Python build system
- `wheel` - For wheel creation
- `setuptools>=60` - Build backend

These will be automatically installed if missing.

## Build Options

### Command Line Arguments

- `--clean` - Remove build artifacts before building
- `--install` - Install the built wheel after building
- `--editable` - Install in editable/development mode
- `--skip-validation` - Skip wheel validation
- `--output-dir DIR` - Specify output directory for wheels

### Build Process

1. **Validation** - Checks project structure and required files
2. **Cleaning** - Removes old build artifacts (if `--clean`)
3. **Dependency Check** - Installs required build tools
4. **Wheel Building** - Creates wheel using `python -m build`
5. **Validation** - Validates the built wheel
6. **Installation** - Installs wheel (if `--install`)

## Output

Built wheels are placed in the `dist/` directory by default:

```
dist/
├── abyss-0.2.5-py3-none-any.whl
└── ...
```

## Project Structure

The build process expects this project structure:

```
abyss/
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
├── src/
│   └── abyss/
│       ├── __init__.py
│       └── ...
├── tests/                  # Test files
└── build_wheel.py         # Build script
```

## Configuration

The build configuration is in `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=60", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "abyss"
version = "0.2.5"
# ... other configuration
```

## Troubleshooting

### Common Issues

1. **Missing build tools**
   ```
   Error: No module named 'build'
   ```
   Solution: The script will automatically install missing packages.

2. **Permission errors**
   ```
   Permission denied: './build_wheel.sh'
   ```
   Solution: Make the script executable with `chmod +x build_wheel.sh`

3. **Import errors after installation**
   ```
   ModuleNotFoundError: No module named 'abyss'
   ```
   Solution: Ensure you're in the correct environment and the package was installed successfully.

### Debugging

Enable verbose output:

```bash
# Python script with verbose pip
python3 build_wheel.py --clean --install -v

# Manual build with verbose output
python3 -m build --wheel --outdir dist . --verbose
```

### Clean Installation

For a completely clean installation:

```bash
# Remove old installations
pip uninstall abyss -y

# Clean build
./build_wheel.sh --clean --install
```

## Development Workflow

### Typical Development Cycle

1. **Make changes** to source code
2. **Run tests** to ensure functionality
   ```bash
   make test
   ```
3. **Build and install** in development mode
   ```bash
   make dev-install
   ```
4. **Test changes** in your environment
5. **Create production build** when ready
   ```bash
   make clean build
   ```

### Testing the Build

```bash
# Build wheel
make build

# Install in a virtual environment
python3 -m venv test_env
source test_env/bin/activate
pip install dist/abyss-*.whl

# Test import
python3 -c "import abyss; print('Success!')"

# Deactivate and cleanup
deactivate
rm -rf test_env
```

## CI/CD Integration

For automated builds, use the Python script directly:

```yaml
# GitHub Actions example
- name: Build wheel
  run: python3 build_wheel.py --clean

- name: Upload wheel
  uses: actions/upload-artifact@v3
  with:
    name: wheel
    path: dist/*.whl
```

## Dependencies

The package dependencies are defined in:

- `requirements.txt` - Runtime dependencies
- `pyproject.toml` - Build system dependencies
- `requirements.cpu` - CPU-only dependencies
- `requirements.docker` - Docker-specific dependencies

Make sure your environment has the appropriate dependencies installed for your use case.