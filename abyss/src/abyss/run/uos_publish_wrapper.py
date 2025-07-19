#!/usr/bin/env python3
"""
MQTT Publisher Wrapper - Compatibility layer for new module structure

This wrapper provides backwards compatibility for existing scripts and
Docker configurations that expect the old publisher scripts.

It now delegates to the new unified publisher module:
    python -m abyss.mqtt.publishers

This ensures zero downtime during the transition to the new structure.
"""

import sys
import subprocess
import logging


def main():
    """Main wrapper function that delegates to the new publisher module"""
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PublisherWrapper")
    
    # Inform about the redirect
    logger.info("Redirecting to new unified publisher module...")
    
    # Build command for the new module
    cmd = [sys.executable, "-m", "abyss.mqtt.publishers"] + sys.argv[1:]
    
    try:
        # Run the new module
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to run new publisher module: {e}")
        logger.error("Falling back to legacy implementation...")
        
        # Fallback to legacy implementation
        try:
            # Check which mode to use
            if '--async' in sys.argv or '--async-mode' in sys.argv:
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
            else:
                # Use normal publisher
                from uos_publish_json import main as normal_main
                normal_main()
        except Exception as legacy_error:
            logger.error(f"Legacy implementation also failed: {legacy_error}")
            sys.exit(1)


if __name__ == "__main__":
    main()