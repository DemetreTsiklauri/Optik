from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
import sys
import numpy as np

def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Optik Hand Control")
    window.resize(400, 300)

    central = QWidget()
    layout = QVBoxLayout(central)
    layout.setAlignment(Qt.AlignCenter)

    status = QLabel("Status: Not tracking")
    status.setAlignment(Qt.AlignCenter)
    btn_start = QPushButton("Start Tracking")
    btn_stop = QPushButton("Stop Tracking")
    btn_stop.setEnabled(False)

    def start():
        try:
            print("Before import cv2")
            import cv2
            print("After import cv2")
            arr = np.array([1, 2, 3, 4, 5])
            result = np.sum(arr)
            print("Before accessing cv2.__version__")
            cv2_version = cv2.__version__
            print("After accessing cv2.__version__")
            status.setText(f"Numpy sum: {result}, OpenCV version: {cv2_version}")
        except Exception as e:
            status.setText(f"OpenCV import failed: {str(e)}")
        btn_start.setEnabled(False)
        btn_stop.setEnabled(True)
    def stop():
        status.setText("Status: Not tracking")
        btn_start.setEnabled(True)
        btn_stop.setEnabled(False)

    btn_start.clicked.connect(start)
    btn_stop.clicked.connect(stop)

    layout.addWidget(status)
    layout.addWidget(btn_start)
    layout.addWidget(btn_stop)
    window.setCentralWidget(central)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 