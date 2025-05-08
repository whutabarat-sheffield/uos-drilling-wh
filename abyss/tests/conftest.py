import sys
import os
from pathlib import Path

print("conftest.py is being loaded")

def pytest_configure(config):
    """Add src directory to Python path before test collection."""
    root_dir = Path(__file__).parent.parent
    src_path = str(root_dir / 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)