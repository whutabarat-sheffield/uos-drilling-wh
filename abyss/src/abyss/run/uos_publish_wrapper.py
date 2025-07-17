#!/usr/bin/env python3
"""
MQTT Publisher Wrapper - Handles both normal and stress test modes

This wrapper script provides a unified interface for both normal publishing
and high-performance stress testing.
"""

import os
import sys
import argparse

# Import both publisher modules
from uos_publish_json import main as normal_main
from uos_publish_json_stress import main as stress_main


def main():
    """Main wrapper function to route to appropriate publisher"""
    # Check if --stress-test is in arguments
    if '--stress-test' in sys.argv:
        # Use stress test publisher
        stress_main()
    else:
        # Use normal publisher
        normal_main()


if __name__ == "__main__":
    main()