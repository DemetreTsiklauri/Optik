#!/usr/bin/env python3
import os
import sys
import traceback
import logging
from main_application import OptikHandControlApp
from PySide6.QtWidgets import QMessageBox

if hasattr(sys, '_MEIPASS'):
    os.chdir(sys._MEIPASS)

sys.stderr = open("/tmp/optik_error.log", "w")
sys.stdout = sys.stderr

def setup_logging():
    """Setup logging configuration"""
    log_dir = os.path.join(os.path.expanduser("~"), "Library", "Logs", "OptikHandControl")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "app.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main entry point"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create and run application
        app = OptikHandControlApp()
        QMessageBox.information(None, "Debug", "App started!")
        return app.run()
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 