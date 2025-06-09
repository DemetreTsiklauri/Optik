# Optik

<img src="assets/optik.png" alt="Optik Logo" width="200"/>

A powerful hand tracking and gesture control system that allows you to control your computer using natural hand movements and voice commands. Built with Python, MediaPipe, and PySide6.

## Features

- Real-time hand tracking with MediaPipe
- Mouse control with hand gestures
- Click and drag functionality
- Scroll control
- Voice dictation
- Enter key simulation
- Smooth cursor movement
- Dynamic edge boosting for precise control

## Requirements

- Python 3.8 or higher
- Webcam
- macOS (for voice dictation support)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/optik.git
cd optik
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Optional Configuration

You can customize the application by creating a `.env` file. This is completely optional - the application will work fine with default settings.

If you want to customize settings, create a `.env` file with any of these variables:

```env
# Camera settings (defaults shown)
CAMERA_ID=0
CAMERA_WIDTH=960
CAMERA_HEIGHT=540

# UI settings (defaults shown)
UI_WIDTH=1280
UI_HEIGHT=720
UI_TITLE="Optik"
UI_FPS=30
```

## Usage

1. Run the application:
```bash
python -m src.main
```

2. Click "Start Tracking" to begin hand tracking.

3. Use the following gestures:
   - Move hand: Control mouse cursor
   - Pinch index and thumb: Click
   - Hold pinch: Drag
   - Pinch index and middle finger: Enter key
   - Extend index and middle: Scroll
   - Extend index, middle, and ring: Scroll up
   - Extend all fingers: Voice dictation

## Project Structure

```
optik/
├── src/
│   ├── main.py           # Main application
│   ├── tracking.py       # Hand tracking
│   ├── smoothing.py      # Movement smoothing
│   └── dictation.py      # Voice dictation
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Dependencies

- MediaPipe for hand tracking
- OpenCV for camera handling
- PySide6 for GUI
- PyAutoGUI for system control
- SpeechRecognition for voice dictation

## License

MIT License
