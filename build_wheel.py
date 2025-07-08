#!/usr/bin/env python3
"""
Wheel Build Script for UOS Drilling Depth Estimation System

This script builds a wheel distribution for the abyss package using pyproject.toml.
It handles the build process, validation, and cleanup operations.

Usage:
    python build_wheel.py [options]

Options:
    --clean         Clean build artifacts before building
    --validate      Validate the built wheel
    --output-dir    Specify output directory for wheels (default: dist/)
    --verbose       Enable verbose output
    --help          Show this help message

Examples:
    python build_wheel.py
    python build_wheel.py --clean --validate
    python build_wheel.py --output-dir=wheels/ --verbose
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional


class WheelBuilder:
    """Handles building wheel distributions for the abyss package."""
    
    def __init__(self, output_dir: str = "dist", verbose: bool = False):
        """
        Initialize WheelBuilder.
        
        Args:
            output_dir: Directory to output built wheels
            verbose: Enable verbose logging
        """
        self.root_dir = Path(__file__).parent
        self.abyss_dir = self.root_dir / "abyss"
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Verify abyss directory exists
        if not self.abyss_dir.exists():
            raise FileNotFoundError(f"Abyss package directory not found: {self.abyss_dir}")
        
        # Verify pyproject.toml exists
        self.pyproject_path = self.abyss_dir / "pyproject.toml"
        if not self.pyproject_path.exists():
            raise FileNotFoundError(f"pyproject.toml not found: {self.pyproject_path}")
    
    def log(self, message: str, force: bool = False):
        """Log message if verbose mode is enabled."""
        if self.verbose or force:
            print(f"[BUILD] {message}")
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        if cwd is None:
            cwd = self.abyss_dir
            
        self.log(f"Running: {' '.join(cmd)} in {cwd}")
        
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        
        if self.verbose and result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print(f"[ERROR] {result.stderr}", file=sys.stderr)
        
        return result
    
    def clean_build_artifacts(self):
        """Clean existing build artifacts."""
        self.log("Cleaning build artifacts...")
        
        # Directories to clean
        clean_dirs = [
            self.abyss_dir / "build",
            self.abyss_dir / "dist",
            self.abyss_dir / "*.egg-info",
            self.output_dir
        ]
        
        for clean_dir in clean_dirs:
            if clean_dir.name.endswith("*.egg-info"):
                # Handle glob pattern for egg-info directories
                for egg_info in clean_dir.parent.glob(clean_dir.name):
                    if egg_info.is_dir():
                        self.log(f"Removing: {egg_info}")
                        shutil.rmtree(egg_info, ignore_errors=True)
            elif clean_dir.exists():
                self.log(f"Removing: {clean_dir}")
                shutil.rmtree(clean_dir, ignore_errors=True)
    
    def ensure_build_dependencies(self):
        """Ensure required build dependencies are installed."""
        self.log("Checking build dependencies...")
        
        required_packages = ["build", "wheel", "setuptools"]
        missing_packages = []
        
        for package in required_packages:
            result = self.run_command([sys.executable, "-c", f"import {package}"])
            if result.returncode != 0:
                missing_packages.append(package)
        
        if missing_packages:
            self.log(f"Installing missing build dependencies: {missing_packages}", force=True)
            result = self.run_command([
                sys.executable, "-m", "pip", "install", "--upgrade"
            ] + missing_packages)
            
            if result.returncode != 0:
                raise RuntimeError("Failed to install build dependencies")
    
    def build_wheel(self) -> Path:
        """Build the wheel distribution."""
        self.log("Building wheel distribution...", force=True)
        
        # Build wheel using python -m build (builds in abyss/dist by default)
        result = self.run_command([
            sys.executable, "-m", "build",
            "--wheel",
            "--outdir", "dist"  # Relative to abyss directory
        ])
        
        if result.returncode != 0:
            raise RuntimeError("Wheel build failed")
        
        # Find the built wheel in abyss/dist (where build puts it)
        abyss_dist_dir = self.abyss_dir / "dist"
        wheel_files = list(abyss_dist_dir.glob("*.whl"))
        if not wheel_files:
            raise RuntimeError("No wheel file found after build")
        
        wheel_path = wheel_files[-1]  # Get the most recent wheel
        self.log(f"Built wheel: {wheel_path}", force=True)
        
        return wheel_path
    
    def validate_wheel(self, wheel_path: Path):
        """Validate the built wheel."""
        self.log("Validating wheel...", force=True)
        
        # Check if wheel file exists and is readable
        if not wheel_path.exists():
            raise RuntimeError(f"Wheel file not found: {wheel_path}")
        
        # Verify wheel is a valid zip file
        try:
            with zipfile.ZipFile(wheel_path, 'r') as wheel_zip:
                file_list = wheel_zip.namelist()
                self.log(f"Wheel contains {len(file_list)} files")
                
                # Check for required files
                required_patterns = [
                    "abyss/",  # Package directory
                    "*.dist-info/",  # Metadata directory
                ]
                
                for pattern in required_patterns:
                    if not any(f.startswith(pattern.replace("*", "")) or pattern.replace("*", "") in f for f in file_list):
                        raise RuntimeError(f"Wheel missing required pattern: {pattern}")
                
                self.log("Wheel structure validation passed")
                
        except zipfile.BadZipFile:
            raise RuntimeError("Built wheel is not a valid zip file")
        
        # Test wheel installation in temporary environment
        self.log("Testing wheel installation...")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_command([
                sys.executable, "-m", "pip", "install",
                "--target", temp_dir,
                "--no-deps",  # Don't install dependencies for validation
                str(wheel_path)
            ])
            
            if result.returncode != 0:
                raise RuntimeError("Wheel installation test failed")
            
            self.log("Wheel installation test passed")
    
    def get_wheel_info(self, wheel_path: Path) -> dict:
        """Extract information from the built wheel."""
        info = {
            "path": str(wheel_path),
            "size": wheel_path.stat().st_size,
            "size_mb": round(wheel_path.stat().st_size / (1024 * 1024), 2)
        }
        
        # Extract version from filename
        filename = wheel_path.name
        parts = filename.split("-")
        if len(parts) >= 2:
            info["package"] = parts[0]
            info["version"] = parts[1]
        
        return info
    
    def build(self, clean: bool = False, validate: bool = False) -> Path:
        """
        Main build process.
        
        Args:
            clean: Whether to clean build artifacts first
            validate: Whether to validate the built wheel
            
        Returns:
            Path to the built wheel file
        """
        try:
            self.log("Starting wheel build process...", force=True)
            
            if clean:
                self.clean_build_artifacts()
            
            self.ensure_build_dependencies()
            wheel_path = self.build_wheel()
            
            if validate:
                self.validate_wheel(wheel_path)
            
            # Display build summary
            info = self.get_wheel_info(wheel_path)
            self.log("=" * 50, force=True)
            self.log("BUILD SUMMARY", force=True)
            self.log("=" * 50, force=True)
            self.log(f"Package: {info.get('package', 'unknown')}", force=True)
            self.log(f"Version: {info.get('version', 'unknown')}", force=True)
            self.log(f"Wheel file: {info['path']}", force=True)
            self.log(f"Size: {info['size_mb']} MB ({info['size']} bytes)", force=True)
            self.log("=" * 50, force=True)
            
            return wheel_path
            
        except Exception as e:
            self.log(f"Build failed: {e}", force=True)
            raise


def main():
    """Main entry point for the build script."""
    parser = argparse.ArgumentParser(
        description="Build wheel distribution for UOS Drilling Depth Estimation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build artifacts before building"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the built wheel"
    )
    
    parser.add_argument(
        "--output-dir",
        default="dist",
        help="Output directory for wheels (default: dist/)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        builder = WheelBuilder(
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        wheel_path = builder.build(
            clean=args.clean,
            validate=args.validate
        )
        
        print(f"\n✓ Successfully built wheel: {wheel_path}")
        return 0
        
    except Exception as e:
        print(f"\n✗ Build failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())