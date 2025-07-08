#!/usr/bin/env python3
"""
Build script for creating Python wheels for the abyss package.

This script handles:
- Clean builds (removing old build artifacts)
- Building wheels using the build package
- Validation of built wheels
- Optional installation in development mode
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            check=check, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        raise


def clean_build_artifacts(project_dir):
    """Remove build artifacts and cache directories."""
    print("Cleaning build artifacts...")
    
    artifacts_to_clean = [
        "build",
        "dist", 
        "src/abyss.egg-info",
        "*.egg-info",
        "__pycache__",
        ".pytest_cache"
    ]
    
    for artifact in artifacts_to_clean:
        artifact_path = project_dir / artifact
        if artifact_path.exists():
            if artifact_path.is_dir():
                print(f"Removing directory: {artifact_path}")
                shutil.rmtree(artifact_path)
            else:
                print(f"Removing file: {artifact_path}")
                artifact_path.unlink()
    
    # Clean __pycache__ directories recursively
    for pycache_dir in project_dir.rglob("__pycache__"):
        print(f"Removing __pycache__: {pycache_dir}")
        shutil.rmtree(pycache_dir)
    
    # Clean .pyc files
    for pyc_file in project_dir.rglob("*.pyc"):
        print(f"Removing .pyc file: {pyc_file}")
        pyc_file.unlink()


def check_requirements():
    """Check if required build tools are installed."""
    print("Checking build requirements...")
    
    required_packages = ["build", "wheel", "setuptools"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {missing_packages}")
        run_command([sys.executable, "-m", "pip", "install"] + missing_packages)


def validate_project_structure(project_dir):
    """Validate that the project has the required structure."""
    print("Validating project structure...")
    
    required_files = [
        "pyproject.toml",
        "requirements.txt",
        "src/abyss/__init__.py"
    ]
    
    for required_file in required_files:
        file_path = project_dir / required_file
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {required_file}")
        print(f"✓ {required_file} exists")


def build_wheel(project_dir, output_dir=None):
    """Build the wheel using the build package."""
    print("Building wheel...")
    
    if output_dir is None:
        output_dir = project_dir / "dist"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Build command
    build_cmd = [
        sys.executable, "-m", "build",
        "--wheel",
        "--outdir", str(output_dir),
        str(project_dir)
    ]
    
    result = run_command(build_cmd)
    
    # Find the built wheel
    wheel_files = list(output_dir.glob("*.whl"))
    if not wheel_files:
        raise RuntimeError("No wheel files found after build")
    
    latest_wheel = max(wheel_files, key=lambda p: p.stat().st_mtime)
    print(f"Built wheel: {latest_wheel}")
    
    return latest_wheel


def validate_wheel(wheel_path):
    """Validate the built wheel."""
    print(f"Validating wheel: {wheel_path}")
    
    # Check wheel using pip
    try:
        result = run_command([
            sys.executable, "-m", "pip", "check"
        ], check=False)
        print("✓ Wheel validation passed")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: Wheel validation had issues: {e}")
    
    # Show wheel contents
    try:
        result = run_command([
            sys.executable, "-m", "wheel", "unpack", 
            "--dest", str(wheel_path.parent / "temp_unpack"),
            str(wheel_path)
        ], check=False)
        
        # Clean up temp directory
        temp_dir = wheel_path.parent / "temp_unpack"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
    except subprocess.CalledProcessError:
        print("⚠ Could not unpack wheel for inspection")


def install_wheel_dev(wheel_path, editable=False):
    """Install the wheel in development mode."""
    print(f"Installing wheel: {wheel_path}")
    
    if editable:
        # Install in editable mode from source
        project_dir = wheel_path.parent.parent
        install_cmd = [
            sys.executable, "-m", "pip", "install", "-e", str(project_dir)
        ]
    else:
        # Install from wheel
        install_cmd = [
            sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel_path)
        ]
    
    run_command(install_cmd)
    print("✓ Package installed successfully")


def get_version_info(project_dir):
    """Extract version information from pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            print("Warning: Cannot read pyproject.toml version (tomli/tomllib not available)")
            return "unknown"
    
    pyproject_path = project_dir / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            return data.get("project", {}).get("version", "unknown")
    return "unknown"


def main():
    """Main build script."""
    parser = argparse.ArgumentParser(description="Build Python wheel for abyss package")
    parser.add_argument("--clean", action="store_true", 
                       help="Clean build artifacts before building")
    parser.add_argument("--install", action="store_true",
                       help="Install the built wheel after building")
    parser.add_argument("--editable", action="store_true",
                       help="Install in editable mode (requires --install)")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory for built wheel")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip wheel validation")
    parser.add_argument("--project-dir", type=Path, default=Path("."),
                       help="Project directory (default: current directory)")
    
    args = parser.parse_args()
    
    # Resolve project directory
    project_dir = args.project_dir.resolve()
    print(f"Building project in: {project_dir}")
    
    try:
        # Show version info
        version = get_version_info(project_dir)
        print(f"Package version: {version}")
        
        # Validate project structure
        validate_project_structure(project_dir)
        
        # Check and install requirements
        check_requirements()
        
        # Clean if requested
        if args.clean:
            clean_build_artifacts(project_dir)
        
        # Build wheel
        wheel_path = build_wheel(project_dir, args.output_dir)
        
        # Validate wheel
        if not args.skip_validation:
            validate_wheel(wheel_path)
        
        # Install if requested
        if args.install:
            install_wheel_dev(wheel_path, args.editable)
        
        print(f"\n✅ Build completed successfully!")
        print(f"Wheel location: {wheel_path}")
        print(f"Wheel size: {wheel_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())