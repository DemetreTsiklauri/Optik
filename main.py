"""Entry point to launch Optik GUI."""
from main_application import OptikHandControlApp
import sys

if __name__ == "__main__":
    app = OptikHandControlApp()
    sys.exit(app.run())
