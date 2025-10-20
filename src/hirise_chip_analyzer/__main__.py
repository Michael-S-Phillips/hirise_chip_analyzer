"""Entry point for HiRISE Chip Analyzer application.

Run this module to launch the GUI application:
    python -m hirise_chip_analyzer
"""

import logging
import sys

from PyQt5.QtWidgets import QApplication

from hirise_chip_analyzer.gui.main_window import MainWindow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the application."""
    logger.info("Starting HiRISE Chip Analyzer")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    logger.info("Application window displayed")
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
