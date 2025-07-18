#!/usr/bin/env python3
"""
MQTT Publisher Wrapper - Handles normal, simple, stress test, and async modes

This wrapper script provides a unified interface for:
- Normal publishing mode (timestamp updating, random order)
- Simple publishing mode (timestamp preserving for exact matching)
- High-performance stress testing (threading)
- Ultra-high-performance async mode (asyncio)
"""

import os
import sys
import argparse


def main():
    """Main wrapper function to route to appropriate publisher"""
    # Check which mode to use
    if '--async' in sys.argv:
        # Use async publisher for ultra-high performance
        try:
            from uos_publish_json_async import main as async_main
            async_main()
        except ImportError as e:
            print(f"Error: Cannot import async publisher: {e}")
            print("Make sure aiomqtt is installed: pip install aiomqtt")
            sys.exit(1)
    elif '--stress-test' in sys.argv:
        # Use threaded stress test publisher
        from uos_publish_json_stress import main as stress_main
        stress_main()
    elif '--simple' in sys.argv or '--exact-match' in sys.argv:
        # Use simple publisher for exact timestamp matching
        from uos_publish_json_simple import main as simple_main
        simple_main()
    else:
        # Use normal publisher
        from uos_publish_json import main as normal_main
        normal_main()


if __name__ == "__main__":
    main()