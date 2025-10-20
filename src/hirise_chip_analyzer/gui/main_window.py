"""Main GUI window for HiRISE Chip Analyzer application.

This module provides the primary user interface for the image classification
application using PyQt5.
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QScrollArea,
    QProgressBar,
    QMessageBox,
    QMenuBar,
    QMenu,
    QStatusBar,
    QGroupBox,
    QGridLayout,
    QSlider,
    QSpinBox,
    QCheckBox,
)
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
from skimage.exposure import equalize_adapthist

from hirise_chip_analyzer.core.classifier import Classifier
from hirise_chip_analyzer.core.file_organizer import (
    ConflictResolution,
    FileOperation,
    FileOrganizer,
)
from hirise_chip_analyzer.core.image_loader import ImageLoader

logger = logging.getLogger(__name__)


class ImageViewer(QLabel):
    """Widget for displaying images with scaling."""

    def __init__(self, max_width: int = 600, max_height: int = 600):
        """Initialize image viewer.

        Parameters
        ----------
        max_width : int
            Maximum display width (default: 600)
        max_height : int
            Maximum display height (default: 600)
        """
        super().__init__()
        self.max_width = max_width
        self.max_height = max_height
        self.setAlignment(Qt.AlignCenter)
        self.setText("No image loaded")
        self.setMinimumSize(400, 400)
        self._original_image_data: Optional[np.ndarray] = None

    def display_image(
        self,
        image_array: np.ndarray,
        brightness: int = 0,
        contrast: int = 100,
        sharpening: int = 0,
        clahe_enabled: bool = False,
        invert_enabled: bool = False,
    ) -> None:
        """Display a numpy image array with improved contrast stretching.

        Uses percentile-based stretching (2%-98%) which is standard for
        remote sensing imagery and provides better contrast than min-max.

        Parameters
        ----------
        image_array : np.ndarray
            Image data as numpy array
        brightness : int, optional
            Brightness adjustment (-100 to 100, default: 0)
        contrast : int, optional
            Contrast adjustment (0 to 200, default: 100 = no adjustment)
        sharpening : int, optional
            Sharpening amount (0 to 100, default: 0 = no sharpening)
        clahe_enabled : bool, optional
            Enable CLAHE (Contrast Limited Adaptive Histogram Equalization)
        invert_enabled : bool, optional
            Invert colors (default: False)
        """
        # Store original image for adjustment updates
        self._original_image_data = image_array.copy()

        # Normalize image to 0-255 range with percentile stretching
        if image_array.dtype != np.uint8:
            # Use percentile-based stretching for better contrast
            # This is standard in remote sensing (2%-98% or similar)
            p_low = np.percentile(image_array, 2)
            p_high = np.percentile(image_array, 98)

            if p_high > p_low:
                # Clip to percentile range and scale to 0-255
                clipped = np.clip(image_array, p_low, p_high)
                normalized = (
                    (clipped - p_low) * 255 / (p_high - p_low)
                ).astype(np.uint8)
            else:
                normalized = np.zeros_like(image_array, dtype=np.uint8)
        else:
            normalized = image_array

        # Apply CLAHE if enabled
        if clahe_enabled:
            normalized = self._apply_clahe(normalized)

        # Apply sharpening if enabled
        if sharpening > 0:
            normalized = self._apply_sharpening(normalized, sharpening)

        # Apply brightness and contrast adjustments
        normalized = self._apply_adjustments(
            normalized, brightness, contrast
        )

        # Apply color inversion if enabled
        if invert_enabled:
            normalized = self._apply_invert(normalized)

        # Convert to PIL Image
        if len(normalized.shape) == 2:
            # Grayscale
            pil_image = PILImage.fromarray(normalized, mode='L')
        else:
            # Color image
            pil_image = PILImage.fromarray(normalized)

        # Resize to fit display while maintaining aspect ratio
        pil_image.thumbnail(
            (self.max_width, self.max_height),
            PILImage.Resampling.LANCZOS
        )

        # Convert PIL Image to QPixmap directly (more reliable)
        # This avoids byte conversion issues
        pixmap = QPixmap.fromImage(
            self._pil_to_qimage(pil_image)
        )
        self.setPixmap(pixmap)

    def _pil_to_qimage(self, pil_image: PILImage.Image) -> QImage:
        """Convert PIL Image to QImage reliably.

        Parameters
        ----------
        pil_image : PIL.Image
            PIL image to convert

        Returns
        -------
        qimage : QImage
            Converted QImage
        """
        if pil_image.mode == 'L':
            # Grayscale
            data = pil_image.tobytes()
            qimage = QImage(
                data,
                pil_image.width,
                pil_image.height,
                pil_image.width,
                QImage.Format_Grayscale8
            )
        elif pil_image.mode == 'RGB':
            # RGB color
            data = pil_image.tobytes()
            qimage = QImage(
                data,
                pil_image.width,
                pil_image.height,
                3 * pil_image.width,
                QImage.Format_RGB888
            )
        else:
            # Convert any other format to RGB
            rgb_image = pil_image.convert('RGB')
            data = rgb_image.tobytes()
            qimage = QImage(
                data,
                rgb_image.width,
                rgb_image.height,
                3 * rgb_image.width,
                QImage.Format_RGB888
            )
        return qimage

    def _apply_adjustments(
        self,
        image_data: np.ndarray,
        brightness: int,
        contrast: int,
    ) -> np.ndarray:
        """Apply brightness and contrast adjustments to image.

        Parameters
        ----------
        image_data : np.ndarray
            Image array to adjust
        brightness : int
            Brightness adjustment (-100 to 100)
        contrast : int
            Contrast adjustment (0 to 200, where 100 = no adjustment)

        Returns
        -------
        adjusted : np.ndarray
            Adjusted image array
        """
        # Convert to float for calculations
        adjusted = image_data.astype(np.float32)

        # Apply contrast adjustment
        # contrast factor: 0.5 at value 0, 1.0 at value 100, 2.0 at value 200
        if contrast != 100:
            contrast_factor = contrast / 100.0
            midpoint = 128.0
            adjusted = (adjusted - midpoint) * contrast_factor + midpoint

        # Apply brightness adjustment
        # brightness range: -100 to 100 maps to -127.5 to 127.5
        if brightness != 0:
            brightness_offset = (brightness / 100.0) * 127.5
            adjusted = adjusted + brightness_offset

        # Clamp to 0-255 range and convert back to uint8
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        return adjusted

    def _apply_clahe(self, image_data: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        This technique enhances local contrast and makes small features like
        pits more visible. It's commonly used in remote sensing.

        Parameters
        ----------
        image_data : np.ndarray
            Grayscale image array (0-255)

        Returns
        -------
        enhanced : np.ndarray
            CLAHE-enhanced image array
        """
        # Normalize to 0-1 range for skimage
        normalized = image_data.astype(np.float32) / 255.0

        # Apply CLAHE with typical parameters for remote sensing
        enhanced = equalize_adapthist(
            normalized,
            kernel_size=50,
            clip_limit=0.03,
            nbins=256
        )

        # Convert back to 0-255 range
        enhanced = (enhanced * 255).astype(np.uint8)
        return enhanced

    def _apply_sharpening(
        self,
        image_data: np.ndarray,
        strength: int,
    ) -> np.ndarray:
        """Apply unsharp masking sharpening to image.

        Makes edges and small features more pronounced.

        Parameters
        ----------
        image_data : np.ndarray
            Image array to sharpen
        strength : int
            Sharpening strength (0-100)

        Returns
        -------
        sharpened : np.ndarray
            Sharpened image array
        """
        # Normalize strength to reasonable range
        sigma = 1.0  # Gaussian blur sigma
        amount = 0.5 + (strength / 100.0) * 1.5  # Range from 0.5 to 2.0

        # Create Gaussian blur
        blurred = gaussian_filter(image_data.astype(np.float32), sigma)

        # Unsharp mask: original + (original - blurred) * amount
        sharpened = image_data.astype(np.float32) + (
            image_data.astype(np.float32) - blurred
        ) * amount

        # Clamp to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened

    def _apply_invert(self, image_data: np.ndarray) -> np.ndarray:
        """Invert image colors.

        Dark features become light and vice versa.

        Parameters
        ----------
        image_data : np.ndarray
            Image array to invert

        Returns
        -------
        inverted : np.ndarray
            Color-inverted image array
        """
        return 255 - image_data


class MainWindow(QMainWindow):
    """Main application window for HiRISE Chip Analyzer."""

    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.setWindowTitle("HiRISE Chip Analyzer - Image Classification Tool")
        self.setGeometry(100, 100, 1000, 800)

        # Initialize core components
        self.image_loader = ImageLoader()
        self.classifier = Classifier()
        self.file_organizer = FileOrganizer()

        # State variables
        self.source_directory: Optional[Path] = None
        self.output_directory: Optional[Path] = None
        self.current_images: list = []
        self.current_index: int = 0
        self.total_images_loaded: int = 0  # Track original count for statistics
        self.classifications = ["pits", "no_pits"]

        # Image adjustment parameters
        self.brightness: int = 0  # Range: -100 to 100
        self.contrast: int = 100  # Range: 0 to 200 (100 = no adjustment)
        self.sharpening: int = 0  # Range: 0 to 100 (0 = no sharpening)
        self.clahe_enabled: bool = False  # Contrast Limited Adaptive Histogram Equalization
        self.invert_enabled: bool = False  # Invert colors

        # Create UI
        self._create_menu_bar()
        self._create_central_widget()
        self._create_status_bar()

        logger.info("Application started")

    def _create_menu_bar(self) -> None:
        """Create application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = file_menu.addAction("Open Image Directory")
        open_action.triggered.connect(self._open_source_directory)

        set_output_action = file_menu.addAction("Set Output Directory")
        set_output_action.triggered.connect(self._set_output_directory)

        file_menu.addSeparator()

        save_session_action = file_menu.addAction("Save Session")
        save_session_action.triggered.connect(self._save_session)

        load_session_action = file_menu.addAction("Load Session")
        load_session_action.triggered.connect(self._load_session)

        file_menu.addSeparator()

        export_summary_action = file_menu.addAction("Export Summary")
        export_summary_action.triggered.connect(self._export_summary)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self._show_about)

    def _create_central_widget(self) -> None:
        """Create main central widget with layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()

        # Directory selection section
        dir_group = QGroupBox("Configuration")
        dir_layout = QGridLayout()

        self.source_label = QLabel("Source Directory: Not selected")
        self.source_button = QPushButton("Select Source Directory")
        self.source_button.clicked.connect(self._open_source_directory)
        dir_layout.addWidget(self.source_label, 0, 0)
        dir_layout.addWidget(self.source_button, 0, 1)

        self.output_label = QLabel("Output Directory: Not selected")
        self.output_button = QPushButton("Select Output Directory")
        self.output_button.clicked.connect(self._set_output_directory)
        dir_layout.addWidget(self.output_label, 1, 0)
        dir_layout.addWidget(self.output_button, 1, 1)

        dir_group.setLayout(dir_layout)
        main_layout.addWidget(dir_group)

        # Image display section
        image_group = QGroupBox("Current Image")
        image_layout = QVBoxLayout()

        self.image_viewer = ImageViewer()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_viewer)
        scroll_area.setWidgetResizable(True)
        image_layout.addWidget(scroll_area)

        self.image_label = QLabel("No image loaded")
        image_layout.addWidget(self.image_label)

        image_group.setLayout(image_layout)

        # Image adjustment section (vertical sliders for right side)
        adjustment_group = QGroupBox("Image Adjustment")
        adjustment_layout = QVBoxLayout()

        # Brightness slider
        brightness_label_layout = QHBoxLayout()
        brightness_label_layout.addWidget(QLabel("Brightness:"))
        self.brightness_value = QLabel("0")
        brightness_label_layout.addWidget(self.brightness_value)
        adjustment_layout.addLayout(brightness_label_layout)

        self.brightness_slider = QSlider(Qt.Vertical)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickPosition(QSlider.TicksLeft)
        self.brightness_slider.setTickInterval(25)
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        self.brightness_slider.setMinimumHeight(150)
        adjustment_layout.addWidget(self.brightness_slider)

        # Contrast slider
        contrast_label_layout = QHBoxLayout()
        contrast_label_layout.addWidget(QLabel("Contrast:"))
        self.contrast_value = QLabel("100")
        contrast_label_layout.addWidget(self.contrast_value)
        adjustment_layout.addLayout(contrast_label_layout)

        self.contrast_slider = QSlider(Qt.Vertical)
        self.contrast_slider.setMinimum(0)
        self.contrast_slider.setMaximum(200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setTickPosition(QSlider.TicksLeft)
        self.contrast_slider.setTickInterval(25)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        self.contrast_slider.setMinimumHeight(150)
        adjustment_layout.addWidget(self.contrast_slider)

        # Sharpening slider
        sharpening_label_layout = QHBoxLayout()
        sharpening_label_layout.addWidget(QLabel("Sharpen:"))
        self.sharpening_value = QLabel("0")
        sharpening_label_layout.addWidget(self.sharpening_value)
        adjustment_layout.addLayout(sharpening_label_layout)

        self.sharpening_slider = QSlider(Qt.Vertical)
        self.sharpening_slider.setMinimum(0)
        self.sharpening_slider.setMaximum(100)
        self.sharpening_slider.setValue(0)
        self.sharpening_slider.setTickPosition(QSlider.TicksLeft)
        self.sharpening_slider.setTickInterval(25)
        self.sharpening_slider.valueChanged.connect(self._on_sharpening_changed)
        self.sharpening_slider.setMinimumHeight(100)
        adjustment_layout.addWidget(self.sharpening_slider)

        # CLAHE toggle
        self.clahe_checkbox = QCheckBox("CLAHE")
        self.clahe_checkbox.setChecked(False)
        self.clahe_checkbox.stateChanged.connect(self._on_clahe_toggled)
        adjustment_layout.addWidget(self.clahe_checkbox)

        # Invert toggle
        self.invert_checkbox = QCheckBox("Invert")
        self.invert_checkbox.setChecked(False)
        self.invert_checkbox.stateChanged.connect(self._on_invert_toggled)
        adjustment_layout.addWidget(self.invert_checkbox)

        # Reset button
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._reset_adjustments)
        adjustment_layout.addWidget(reset_button)

        # Add stretch to push controls to top
        adjustment_layout.addStretch()

        adjustment_group.setLayout(adjustment_layout)
        adjustment_group.setMaximumWidth(120)

        # Horizontal layout to contain image and adjustments side by side
        image_adjustment_layout = QHBoxLayout()
        image_adjustment_layout.addWidget(image_group, 1)  # Image takes most space
        image_adjustment_layout.addWidget(adjustment_group, 0)  # Sliders on right

        # Create container for image and adjustments
        image_adjustment_container = QWidget()
        image_adjustment_container.setLayout(image_adjustment_layout)
        main_layout.addWidget(image_adjustment_container, 1)

        # Navigation buttons
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout()

        self.prev_button = QPushButton("← Previous")
        self.prev_button.clicked.connect(self._previous_image)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)

        self.skip_button = QPushButton("Skip (No Classification)")
        self.skip_button.clicked.connect(self._skip_image)
        self.skip_button.setEnabled(False)
        nav_layout.addWidget(self.skip_button)

        self.next_button = QPushButton("Next →")
        self.next_button.clicked.connect(self._next_image)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)

        nav_group.setLayout(nav_layout)
        main_layout.addWidget(nav_group)

        # Classification buttons
        button_group = QGroupBox("Classification")
        button_layout = QHBoxLayout()

        self.pits_button = QPushButton("Yes - Pits")
        self.pits_button.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        self.pits_button.clicked.connect(self._classify_pits)
        self.pits_button.setEnabled(False)
        button_layout.addWidget(self.pits_button)

        self.no_pits_button = QPushButton("No - No Pits")
        self.no_pits_button.setStyleSheet("background-color: #FFB6C6; font-weight: bold;")
        self.no_pits_button.clicked.connect(self._classify_no_pits)
        self.no_pits_button.setEnabled(False)
        button_layout.addWidget(self.no_pits_button)

        button_group.setLayout(button_layout)
        main_layout.addWidget(button_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.stats_label = QLabel("Ready. Select a directory to begin.")
        progress_layout.addWidget(self.stats_label)

        progress_group.setLayout(progress_layout)
        main_layout.addWidget(progress_group)

        central_widget.setLayout(main_layout)

    def _create_status_bar(self) -> None:
        """Create status bar."""
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Ready")

    def _open_source_directory(self) -> None:
        """Open dialog to select source image directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Source Image Directory"
        )

        if directory:
            self.source_directory = Path(directory)
            self.source_label.setText(
                f"Source Directory: {self.source_directory.name}"
            )
            self.statusbar.showMessage(
                f"Loading images from {self.source_directory.name}..."
            )
            self._load_images()

    def _set_output_directory(self) -> None:
        """Open dialog to select output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )

        if directory:
            self.output_directory = Path(directory)
            self.output_label.setText(
                f"Output Directory: {self.output_directory.name}"
            )

            # Create classification subdirectories
            try:
                self.file_organizer.create_classification_directories(
                    self.output_directory,
                    self.classifications
                )
                self.statusbar.showMessage(
                    f"Output directory configured: {self.output_directory.name}"
                )
            except Exception as e:
                QMessageBox.warning(
                    self, "Error",
                    f"Failed to create directories: {str(e)}"
                )
                logger.error(f"Failed to create directories: {e}")

    def _load_images(self) -> None:
        """Load images from source directory."""
        if not self.source_directory:
            return

        try:
            self.current_images = (
                self.image_loader.load_images_from_directory(
                    self.source_directory
                )
            )
            self.current_index = 0
            self.total_images_loaded = len(self.current_images)

            if self.current_images:
                self.pits_button.setEnabled(True)
                self.no_pits_button.setEnabled(True)
                self.next_button.setEnabled(True)
                self.prev_button.setEnabled(False)  # Can't go back from first
                self.skip_button.setEnabled(True)
                self._display_current_image()
                self.statusbar.showMessage(
                    f"Loaded {len(self.current_images)} images"
                )
            else:
                QMessageBox.warning(
                    self, "No Images",
                    "No TIFF images found in the selected directory"
                )
                self.statusbar.showMessage("No images found")

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load images: {str(e)}"
            )
            logger.error(f"Failed to load images: {e}")

    def _display_current_image(self) -> None:
        """Display the current image and update button states."""
        if not self.current_images or self.current_index >= len(
            self.current_images
        ):
            return

        image_data, metadata = self.current_images[self.current_index]

        self.image_viewer.display_image(
            image_data,
            brightness=self.brightness,
            contrast=self.contrast,
            sharpening=self.sharpening,
            clahe_enabled=self.clahe_enabled,
            invert_enabled=self.invert_enabled
        )

        total = self.total_images_loaded
        remaining = len(self.current_images)
        classified_count = total - remaining

        # Progress based on actual classifications
        progress = int(
            100 * classified_count / total
            if total > 0 else 0
        )
        self.progress_bar.setValue(progress)

        self.image_label.setText(
            f"Image {self.current_index + 1} of {remaining} remaining: "
            f"{metadata['filename']}"
        )

        # Update statistics with remaining count
        stats = self.classifier.get_statistics()
        self.stats_label.setText(
            f"Remaining: {remaining} | "
            f"Classified: {classified_count} | "
            f"Pits: {stats['classifications'].get('pits', 0)} | "
            f"No Pits: {stats['classifications'].get('no_pits', 0)}"
        )

        # Update navigation button states
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(
            self.current_index < len(self.current_images) - 1
        )

    def _classify_pits(self) -> None:
        """Classify current image as containing pits."""
        self._classify_and_save("pits")

    def _classify_no_pits(self) -> None:
        """Classify current image as not containing pits."""
        self._classify_and_save("no_pits")

    def _classify_and_save(self, classification: str) -> None:
        """Classify current image and organize file.

        Parameters
        ----------
        classification : str
            Classification label
        """
        # Check if output directory is set
        if not self.output_directory:
            QMessageBox.warning(
                self, "Output Directory Required",
                "Please select an output directory before classifying images.\n\n"
                "Use 'Select Output Directory' button at the top."
            )
            return

        if not self.current_images or self.current_index >= len(
            self.current_images
        ):
            return

        image_data, metadata = self.current_images[self.current_index]
        filename = metadata['filename']

        # Record classification
        try:
            self.classifier.classify_image(
                filename, classification
            )

            # Organize file - output directory is guaranteed to be set
            source_file = (
                self.source_directory / filename
            )
            try:
                self.file_organizer.organize_classification(
                    source_file,
                    classification,
                    self.output_directory,
                    operation=FileOperation.MOVE,
                    conflict_resolution=ConflictResolution.RENAME
                )
            except Exception as e:
                logger.warning(
                    f"Could not move file {filename}: {e}"
                )

            logger.info(
                f"Classified {filename} as {classification}"
            )

        except Exception as e:
            QMessageBox.warning(
                self, "Error",
                f"Failed to classify image: {str(e)}"
            )
            logger.error(f"Failed to classify image: {e}")
            return

        # REMOVE the classified image from the viewing list
        self.current_images.pop(self.current_index)

        # Adjust current_index if we removed the last image in the list
        if self.current_index >= len(self.current_images) and len(
            self.current_images
        ) > 0:
            self.current_index = len(self.current_images) - 1

        # Check if all images have been classified (list is now empty)
        if len(self.current_images) == 0:
            self._show_completion_dialog()
        else:
            self._display_current_image()

    def _show_completion_dialog(self) -> None:
        """Show dialog when all images are classified."""
        self.pits_button.setEnabled(False)
        self.no_pits_button.setEnabled(False)

        stats = self.classifier.get_statistics()

        message = (
            f"All images classified!\n\n"
            f"Total: {stats['total_classified']}\n"
            f"Pits: {stats['classifications'].get('pits', 0)}\n"
            f"No Pits: {stats['classifications'].get('no_pits', 0)}"
        )

        QMessageBox.information(self, "Classification Complete", message)

        self.image_viewer.setText("All images classified!")
        self.image_label.setText("Session complete")

    def _save_session(self) -> None:
        """Save classification session to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                self.classifier.save_session(Path(file_path))
                QMessageBox.information(
                    self, "Success",
                    f"Session saved to {Path(file_path).name}"
                )
                self.statusbar.showMessage("Session saved")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to save session: {str(e)}"
                )
                logger.error(f"Failed to save session: {e}")

    def _load_session(self) -> None:
        """Load classification session from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                self.classifier = Classifier.load_session(Path(file_path))
                QMessageBox.information(
                    self, "Success",
                    f"Session loaded: {len(self.classifier.get_records())} "
                    f"classifications"
                )
                self.statusbar.showMessage("Session loaded")
                self._display_current_image()
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to load session: {str(e)}"
                )
                logger.error(f"Failed to load session: {e}")

    def _export_summary(self) -> None:
        """Export classification summary."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary", "",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                self.classifier.export_summary(Path(file_path))
                QMessageBox.information(
                    self, "Success",
                    f"Summary exported to {Path(file_path).name}"
                )
                self.statusbar.showMessage("Summary exported")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to export summary: {str(e)}"
                )
                logger.error(f"Failed to export summary: {e}")

    def _on_brightness_changed(self, value: int) -> None:
        """Handle brightness slider change.

        Parameters
        ----------
        value : int
            New brightness value (-100 to 100)
        """
        self.brightness = value
        self.brightness_value.setText(str(value))
        self._update_image_display()

    def _on_contrast_changed(self, value: int) -> None:
        """Handle contrast slider change.

        Parameters
        ----------
        value : int
            New contrast value (0 to 200)
        """
        self.contrast = value
        self.contrast_value.setText(str(value))
        self._update_image_display()

    def _on_sharpening_changed(self, value: int) -> None:
        """Handle sharpening slider change.

        Parameters
        ----------
        value : int
            New sharpening value (0 to 100)
        """
        self.sharpening = value
        self.sharpening_value.setText(str(value))
        self._update_image_display()

    def _on_clahe_toggled(self, state: int) -> None:
        """Handle CLAHE toggle change.

        Parameters
        ----------
        state : int
            Qt CheckState (0 = unchecked, 2 = checked)
        """
        from PyQt5.QtCore import Qt
        self.clahe_enabled = state == Qt.Checked
        self._update_image_display()

    def _on_invert_toggled(self, state: int) -> None:
        """Handle invert colors toggle change.

        Parameters
        ----------
        state : int
            Qt CheckState (0 = unchecked, 2 = checked)
        """
        from PyQt5.QtCore import Qt
        self.invert_enabled = state == Qt.Checked
        self._update_image_display()

    def _reset_adjustments(self) -> None:
        """Reset all adjustments to default values."""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.sharpening_slider.setValue(0)
        self.clahe_checkbox.setChecked(False)
        self.invert_checkbox.setChecked(False)
        self.brightness = 0
        self.contrast = 100
        self.sharpening = 0
        self.clahe_enabled = False
        self.invert_enabled = False
        self._update_image_display()

    def _update_image_display(self) -> None:
        """Update image display with current adjustments."""
        if not self.current_images or self.current_index >= len(
            self.current_images
        ):
            return

        image_data, metadata = self.current_images[self.current_index]
        self.image_viewer.display_image(
            image_data,
            brightness=self.brightness,
            contrast=self.contrast,
            sharpening=self.sharpening,
            clahe_enabled=self.clahe_enabled,
            invert_enabled=self.invert_enabled
        )

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self, "About HiRISE Chip Analyzer",
            "HiRISE Chip Analyzer v0.1.0\n\n"
            "A GUI application for classifying HiRISE image chips\n"
            "for the presence of pit features.\n\n"
            "© 2025 HiRISE Team"
        )

    def _next_image(self) -> None:
        """Move to the next image without classifying."""
        if self.current_index < len(self.current_images) - 1:
            self.current_index += 1
            self._display_current_image()
            self.statusbar.showMessage(
                f"Viewing image {self.current_index + 1} of "
                f"{len(self.current_images)}"
            )

    def _previous_image(self) -> None:
        """Move to the previous image without classifying."""
        if self.current_index > 0:
            self.current_index -= 1
            self._display_current_image()
            self.statusbar.showMessage(
                f"Viewing image {self.current_index + 1} of "
                f"{len(self.current_images)}"
            )

    def _skip_image(self) -> None:
        """Skip the current image without classifying and move to next."""
        if self.current_index < len(self.current_images) - 1:
            self.current_index += 1
            self._display_current_image()
            self.statusbar.showMessage(
                f"Image skipped. Now viewing image {self.current_index + 1} "
                f"of {len(self.current_images)}"
            )
