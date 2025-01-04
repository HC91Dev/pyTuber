# main.py
import sys
import os
import logging
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

# Setup logging
logging.basicConfig(
    filename="vrm_tracker.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

if __name__ == "__main__":
    try:
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/qt/plugins'
        
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
        raise