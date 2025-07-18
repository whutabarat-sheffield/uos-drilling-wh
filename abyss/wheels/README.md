# Wheel Dependencies

This directory contains pre-built wheel files for critical ML dependencies that are required for the UOS Drilling depth estimation system.

## Included Wheels

- **accelerate-1.2.1-py3-none-any.whl** - Hugging Face's distributed training and mixed precision library
- **transformers-4.41.0.dev0-py3-none-any.whl** - Development version of Hugging Face Transformers
- **tsfm_public-0.2.17-py3-none-any.whl** - Time Series Foundation Models library

## Why Vendored Wheels?

These wheels are included in the repository for several reasons:

1. **Specific Development Versions**: The transformers library is a dev version (4.41.0.dev0) that may not be available on PyPI
2. **Model Compatibility**: The depth estimation model was trained with these exact versions
3. **Reproducibility**: Ensures all environments use identical library versions
4. **Corporate Environments**: Some deployment environments may have restricted internet access
5. **Performance**: Pre-built wheels avoid compilation time during installation

## Installation

### In Development (devcontainer)
The devcontainer.json automatically installs these wheels using:
```bash
pip install --user --find-links wheels accelerate==1.2.1 transformers==4.41.0.dev0 tsfm_public==0.2.17
```

### Manual Installation
From the abyss directory:
```bash
pip install wheels/*.whl
```

Or using the requirements file:
```bash
pip install -r wheels/requirements.txt
```

### In Docker
The Dockerfile uses requirements.docker which references these wheels:
```bash
pip install -r requirements.docker
```

## Updating Wheels

⚠️ **CAUTION**: These wheels are tightly coupled with the trained ML model. Do not update without:
1. Testing with the existing model
2. Verifying prediction accuracy remains consistent
3. Updating all related requirements files
4. Testing across all deployment environments

## Version Compatibility

- Python: 3.10.x (as specified in devcontainer.json)
- PyTorch: 2.3.1 (must not exceed this version)
- Platform: These are pure Python wheels (platform-independent)

## Security Considerations

These wheels should be:
- Scanned for vulnerabilities regularly
- Updated when security patches are available
- Verified for integrity (consider adding checksums)

## Related Files

- `/abyss/requirements.txt` - Main requirements (wheels commented out)
- `/abyss/requirements.docker` - Docker requirements (includes wheels)
- `/abyss/requirements.devcontainer` - Consider creating for devcontainer-specific needs

## Troubleshooting

If wheel installation fails:
1. Ensure you're in the correct directory (`/abyss`)
2. Check Python version compatibility (3.10.x)
3. Verify wheel files are not corrupted
4. Try installing individually to identify problematic wheel
5. Check for conflicting versions in requirements.txt

---

*Note: This dependency management approach is temporary. Future versions should use proper package management with a private PyPI repository or similar solution.*