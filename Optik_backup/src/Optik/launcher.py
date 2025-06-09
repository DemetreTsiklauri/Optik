#!/usr/bin/env python3
import os
import sys
import traceback

def main():
    try:
        # Add the parent directory to Python path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        
        # Import and run the main app
        from src.optik_gui import main
        main()
    except Exception as e:
        # Log any errors to a file
        with open("/tmp/optik_launcher_error.log", "w") as f:
            f.write(f"Error in launcher: {str(e)}\n")
            traceback.print_exc(file=f)
        raise

if __name__ == "__main__":
    main() 