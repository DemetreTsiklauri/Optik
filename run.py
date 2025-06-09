#!/usr/bin/env python3
import sys
import os
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/optik.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add the Optik package to the Python path
sys.path.append(os.path.abspath("Optik/src"))

try:
    from Optik.optik_gui import main
    logger.info("Starting Optik Hand Control...")
    main()
except Exception as e:
    logger.error(f"Fatal error: {str(e)}")
    traceback.print_exc()
    sys.exit(1) 