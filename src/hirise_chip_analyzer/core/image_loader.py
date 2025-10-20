"""Image loading and management for TIFF image chips.

This module provides functionality to load, validate, and manage HiRISE image
chips in TIFF format. It handles file I/O with proper error handling and
validation of image data. Uses rasterio for GIS-format TIFF files with PIL
fallback for standard TIFF formats.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image

logger = logging.getLogger(__name__)


class ImageLoader:
    """Load and manage HiRISE image chips from TIFF files.

    This class provides methods to load TIFF image files, retrieve image
    metadata, validate image data quality, and manage image collections
    from directories.

    Attributes
    ----------
    supported_formats : tuple
        File extensions supported by the loader (e.g., '.tiff', '.tif')
    """

    supported_formats = ('.tiff', '.tif', '.TIFF', '.TIF')

    @staticmethod
    def load_image(
        image_path: Path,
    ) -> Tuple[np.ndarray, dict]:
        """Load a single TIFF image file.

        Attempts to load using rasterio (for GIS-format TIFF) first,
        then falls back to PIL for standard TIFF formats.

        Parameters
        ----------
        image_path : Path
            Path to the TIFF image file to load

        Returns
        -------
        image_data : np.ndarray
            Image array with shape (height, width) for grayscale or
            (height, width, channels) for multi-channel images
        metadata : dict
            Image metadata including:
            - 'filename': str - Original filename
            - 'shape': tuple - Image dimensions
            - 'dtype': str - Data type (e.g., 'uint8', 'uint16')
            - 'bands': int - Number of color bands/channels
            - 'size_bytes': int - File size in bytes

        Raises
        ------
        FileNotFoundError
            If the image file does not exist
        ValueError
            If the file is not a valid TIFF image
        """
        image_path = Path(image_path)

        if not image_path.exists():
            msg = f"Image file not found: {image_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Try rasterio first (better for GIS/remote sensing TIFF)
        try:
            with rasterio.open(image_path) as src:
                # Read all bands
                image_data = src.read()

                # If single band, reshape from (1, H, W) to (H, W)
                if image_data.shape[0] == 1:
                    image_data = image_data[0]
                else:
                    # Multiple bands: transpose to (H, W, B)
                    image_data = np.transpose(image_data, (1, 2, 0))

                metadata = {
                    'filename': image_path.name,
                    'shape': image_data.shape,
                    'dtype': str(image_data.dtype),
                    'bands': src.count,
                    'size_bytes': image_path.stat().st_size,
                }

                logger.debug(
                    f"Loaded image {image_path.name} (rasterio): "
                    f"shape={metadata['shape']}, "
                    f"dtype={metadata['dtype']}"
                )

                return image_data, metadata

        except Exception as rasterio_error:
            # Fallback to PIL for standard TIFF files
            logger.debug(
                f"Rasterio failed for {image_path.name}, "
                f"trying PIL: {str(rasterio_error)}"
            )

            try:
                with Image.open(image_path) as img:
                    # Convert to numpy array
                    image_data = np.array(img)

                    # Extract metadata
                    metadata = {
                        'filename': image_path.name,
                        'shape': image_data.shape,
                        'dtype': str(image_data.dtype),
                        'bands': image_data.shape[2]
                        if len(image_data.shape) > 2
                        else 1,
                        'size_bytes': image_path.stat().st_size,
                    }

                    logger.debug(
                        f"Loaded image {image_path.name} (PIL): "
                        f"shape={metadata['shape']}, "
                        f"dtype={metadata['dtype']}"
                    )

                    return image_data, metadata

            except Exception as pil_error:
                msg = (
                    f"Failed to load {image_path.name} with both rasterio "
                    f"and PIL. Rasterio error: {str(rasterio_error)}. "
                    f"PIL error: {str(pil_error)}"
                )
                logger.error(msg)
                raise ValueError(msg) from pil_error

    @staticmethod
    def load_images_from_directory(
        directory: Path,
        recursive: bool = False,
    ) -> List[Tuple[np.ndarray, dict]]:
        """Load all TIFF images from a directory.

        Parameters
        ----------
        directory : Path
            Path to directory containing TIFF images
        recursive : bool, optional
            If True, search subdirectories recursively (default: False)

        Returns
        -------
        images : list of tuple
            List of (image_data, metadata) tuples for each loaded image

        Raises
        ------
        FileNotFoundError
            If the directory does not exist
        """
        directory = Path(directory)

        if not directory.exists():
            msg = f"Directory not found: {directory}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not directory.is_dir():
            msg = f"Path is not a directory: {directory}"
            logger.error(msg)
            raise NotADirectoryError(msg)

        # Find all TIFF files
        pattern = "**/*" if recursive else "*"
        image_files = []

        for ext in ImageLoader.supported_formats:
            image_files.extend(directory.glob(f"{pattern}{ext}"))

        if not image_files:
            logger.warning(
                f"No TIFF images found in {directory} "
                f"(recursive={recursive})"
            )
            return []

        logger.info(f"Found {len(image_files)} TIFF files in {directory}")

        # Load images
        images = []
        for image_path in sorted(image_files):
            try:
                image_data, metadata = ImageLoader.load_image(image_path)
                images.append((image_data, metadata))
            except Exception as e:
                logger.warning(
                    f"Failed to load image {image_path.name}: {str(e)}"
                )
                continue

        logger.info(f"Successfully loaded {len(images)} images")
        return images

    @staticmethod
    def get_image_filenames(
        directory: Path,
        recursive: bool = False,
    ) -> List[Path]:
        """Get list of TIFF image file paths in a directory.

        Parameters
        ----------
        directory : Path
            Path to directory to search
        recursive : bool, optional
            If True, search subdirectories recursively (default: False)

        Returns
        -------
        file_paths : list of Path
            Sorted list of Path objects for all TIFF files found

        Raises
        ------
        FileNotFoundError
            If the directory does not exist
        """
        directory = Path(directory)

        if not directory.exists():
            msg = f"Directory not found: {directory}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not directory.is_dir():
            msg = f"Path is not a directory: {directory}"
            logger.error(msg)
            raise NotADirectoryError(msg)

        # Find all TIFF files
        pattern = "**/*" if recursive else "*"
        image_files = []

        for ext in ImageLoader.supported_formats:
            image_files.extend(directory.glob(f"{pattern}{ext}"))

        return sorted(image_files)

    @staticmethod
    def validate_image(
        image_data: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]] = None,
    ) -> bool:
        """Validate image data quality and structure.

        Parameters
        ----------
        image_data : np.ndarray
            Image array to validate
        expected_shape : tuple of int, optional
            Expected shape of the image. If None, only checks that array
            is valid (default: None)

        Returns
        -------
        is_valid : bool
            True if image is valid, False otherwise

        Raises
        ------
        ValueError
            If image fails validation checks
        """
        # Check that it's an array
        if not isinstance(image_data, np.ndarray):
            msg = "Image data must be a numpy array"
            logger.error(msg)
            raise ValueError(msg)

        # Check dimensions
        if image_data.ndim < 2:
            msg = f"Image must be at least 2D, got {image_data.ndim}D"
            logger.error(msg)
            raise ValueError(msg)

        # Check for NaN or infinite values
        if not np.all(np.isfinite(image_data)):
            n_invalid = np.sum(~np.isfinite(image_data))
            logger.warning(
                f"Image contains {n_invalid} non-finite values "
                f"({100*n_invalid/image_data.size:.2f}%)"
            )

        # Check shape if specified
        if expected_shape is not None:
            if image_data.shape != expected_shape:
                msg = (
                    f"Image shape {image_data.shape} does not match "
                    f"expected shape {expected_shape}"
                )
                logger.error(msg)
                raise ValueError(msg)

        logger.debug(
            f"Image validation passed: "
            f"shape={image_data.shape}, dtype={image_data.dtype}"
        )
        return True

    @staticmethod
    def get_image_stats(
        image_data: np.ndarray,
    ) -> dict:
        """Calculate statistical properties of an image.

        Parameters
        ----------
        image_data : np.ndarray
            Image array to analyze

        Returns
        -------
        stats : dict
            Dictionary containing:
            - 'min': float - Minimum pixel value
            - 'max': float - Maximum pixel value
            - 'mean': float - Mean pixel value
            - 'std': float - Standard deviation
            - 'median': float - Median pixel value
            - 'percentiles': dict - 1st, 5th, 95th, 99th percentiles
        """
        return {
            'min': float(np.min(image_data)),
            'max': float(np.max(image_data)),
            'mean': float(np.mean(image_data)),
            'std': float(np.std(image_data)),
            'median': float(np.median(image_data)),
            'percentiles': {
                '1': float(np.percentile(image_data, 1)),
                '5': float(np.percentile(image_data, 5)),
                '95': float(np.percentile(image_data, 95)),
                '99': float(np.percentile(image_data, 99)),
            },
        }
