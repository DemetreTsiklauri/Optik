from PySide6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget

class Ui_MainWindow:
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("Optik")
        MainWindow.setFixedSize(300, 200)
        self.centralwidget = QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)
        self.layout = QVBoxLayout(self.centralwidget)

        self.startButton = QPushButton("Start Tracking")
        self.stopButton = QPushButton("Stop Tracking")

        self.layout.addWidget(self.startButton)
        self.layout.addWidget(self.stopButton)
