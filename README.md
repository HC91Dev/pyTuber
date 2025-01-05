# VRM Viewer and Face Tracker

A Linux-based VRM model viewer and webcam face tracking application built with Python and OpenGL. This tool allows you to load and view VRM/GLTF models and includes webcam-based face tracking capabilities.

## Features

- Load and view VRM and GLTF 3D models
- Real-time webcam face tracking
- Interactive 3D model viewer with mouse controls
- Convert VRM files to GLTF format
- Model validation and inspection tools

## Requirements

### System Dependencies (Arch Linux)
```bash
sudo pacman -S python-pyqt5 python-opengl python-dlib cmake boost-libs openblas
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository
```bash
git clone [your-repo-url]
cd vrm-viewer
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

### Controls
- Left click + drag: Move model up/down
- Right click + drag: Rotate model
- Mouse wheel: Zoom in/out

### Supported File Formats
- VRM (.vrm)
- GLTF (.gltf)
- GLB (.glb)

## Project Structure
```
project_root/
├── main.py                  # Application entry point
├── gui/
│   ├── main_window.py      # Main window implementation
│   ├── vrm_viewer.py       # OpenGL viewer component
│   └── webcam_tracker.py   # Webcam tracking functionality
└── utils/
    └── model_parser.py     # Model parsing utilities
```

## Known Issues
- Some VRM files with complex shaders might not render correctly
- Webcam tracking requires good lighting conditions for optimal performance

## Contributing

Feel free to open issues or submit pull requests!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using PyQt5 for the GUI
- Uses OpenGL for 3D rendering
- Implements face tracking using OpenCV and dlib
- Supports VRM format for VTuber models
