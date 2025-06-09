#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import traceback
import cv2
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt

# Setup error logging
def log_uncaught_exceptions(ex_cls, ex, tb):
    with open("/tmp/optik_error.log", "a") as f:
        f.write("\n=== Uncaught Exception ===\n")
        traceback.print_exception(ex_cls, ex, tb, file=f)
    sys.__excepthook__(ex_cls, ex, tb)

sys.excepthook = log_uncaught_exceptions
sys.stderr = open("/tmp/optik_error.log", "w")
sys.stdout = sys.stderr

print("Optik app starting...")

try:
    from hand_tracker import HandTracker
    from hand_controller import HandController
    from speech_dictation import SpeechDictation
except Exception as e:
    print(f"Error importing modules: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class OptikApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optik Hand Control")
        self.setWindowIcon(QtGui.QIcon(resource_path("assets/optik.icns")))
        
        # Set window size and position
        self.setFixedSize(400, 300)
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        
        # Create central widget and layout
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Add title label
        title = QtWidgets.QLabel("Optik Hand Control")
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Add status label
        self.status_label = QtWidgets.QLabel("Status: Not Tracking")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #7f8c8d;
                padding: 5px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Add buttons
        self.start_btn = QtWidgets.QPushButton("Start Tracking")
        self.stop_btn = QtWidgets.QPushButton("Stop Tracking")
        self.stop_btn.setEnabled(False)
        
        # Style buttons
        button_style = """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """
        self.start_btn.setStyleSheet(button_style)
        self.stop_btn.setStyleSheet(button_style)
        
        # Add buttons to layout
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        # Add version label
        version = QtWidgets.QLabel("Version 1.0")
        version.setStyleSheet("color: #95a5a6;")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)
        
        self.setCentralWidget(central)
        
        # Connect signals
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        
        # Initialize components
        self.tracker = HandTracker()
        self.controller = HandController()
        self.speech = SpeechDictation()
        
        # Setup timer for frame processing
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)  # ~30 FPS
        self.timer.timeout.connect(self._on_frame)
        
        # Track if we're currently processing
        self.is_tracking = False

    def start_tracking(self):
        self.is_tracking = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Tracking Active")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #27ae60;
                padding: 5px;
            }
        """)
        self.timer.start()

    def stop_tracking(self):
        self.is_tracking = False
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Not Tracking")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #7f8c8d;
                padding: 5px;
            }
        """)

    def _on_frame(self):
        if not self.is_tracking:
            return
            
        frame = self.tracker.next_frame()
        if frame is None:
            return
            
        # Flip frame horizontally for proper mirroring
        frame = cv2.flip(frame, 1)
        
        # Process hand tracking
        lm = self.tracker.get_landmarks(frame)
        if lm:
            self.controller.update(lm)
            self.speech.update(lm)

    def closeEvent(self, event):
        """Handle application close"""
        self.stop_tracking()
        self.controller.close()
        event.accept()

def main():
    try:
        print("Initializing Qt application...")
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle("Fusion")
        
        print("Creating main window...")
        window = OptikApp()
        window.show()
        
        print("Starting event loop...")
        sys.exit(app.exec())
    except Exception as e:
        print(f"Fatal error in main(): {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
