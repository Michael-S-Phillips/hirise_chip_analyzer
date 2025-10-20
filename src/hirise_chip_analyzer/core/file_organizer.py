"""File organization and classification output handling.

This module manages the output of classified images, handling file movement,
copying, and organization into classification-based directories.
"""

import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class FileOperation(Enum):
    """Enumeration of file operations."""

    MOVE = "move"
    COPY = "copy"


class ConflictResolution(Enum):
    """Enumeration of conflict resolution strategies."""

    SKIP = "skip"  # Skip if file exists
    OVERWRITE = "overwrite"  # Overwrite existing file
    RENAME = "rename"  # Rename with suffix (e.g., _1, _2)


class FileOrganizer:
    """Manage organization and output of classified image files.

    This class handles moving or copying classified images to appropriate
    output directories based on user classification decisions.
    """

    @staticmethod
    def ensure_directory_exists(directory: Path) -> Path:
        """Ensure that a directory exists, creating it if necessary.

        Parameters
        ----------
        directory : Path
            Path to directory to create or verify

        Returns
        -------
        directory : Path
            The directory path (created if it didn't exist)

        Raises
        ------
        PermissionError
            If cannot create directory due to permissions
        OSError
            If cannot create directory for other reasons
        """
        directory = Path(directory)

        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
            return directory
        except PermissionError as e:
            msg = f"Permission denied creating directory: {directory}"
            logger.error(msg)
            raise PermissionError(msg) from e
        except OSError as e:
            msg = f"Error creating directory {directory}: {str(e)}"
            logger.error(msg)
            raise

    @staticmethod
    def move_file(
        source_file: Path,
        dest_directory: Path,
        conflict_resolution: ConflictResolution = ConflictResolution.SKIP,
    ) -> Tuple[bool, str]:
        """Move a file to a destination directory.

        Parameters
        ----------
        source_file : Path
            Path to the source file
        dest_directory : Path
            Path to destination directory
        conflict_resolution : ConflictResolution, optional
            How to handle existing files (default: SKIP)

        Returns
        -------
        success : bool
            True if operation succeeded, False otherwise
        message : str
            Status message describing what happened

        Raises
        ------
        FileNotFoundError
            If source file does not exist
        """
        source_file = Path(source_file)
        dest_directory = Path(dest_directory)

        if not source_file.exists():
            msg = f"Source file not found: {source_file}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Ensure destination directory exists
        FileOrganizer.ensure_directory_exists(dest_directory)

        dest_file = dest_directory / source_file.name

        # Handle file conflicts
        if dest_file.exists():
            if conflict_resolution == ConflictResolution.SKIP:
                msg = f"File already exists, skipping: {dest_file}"
                logger.info(msg)
                return False, msg

            elif conflict_resolution == ConflictResolution.OVERWRITE:
                dest_file.unlink()
                logger.debug(f"Overwriting existing file: {dest_file}")

            elif conflict_resolution == ConflictResolution.RENAME:
                dest_file = FileOrganizer._get_unique_filename(dest_file)
                logger.debug(f"Renamed to avoid conflict: {dest_file}")

        try:
            shutil.move(str(source_file), str(dest_file))
            msg = f"Moved {source_file.name} to {dest_directory}"
            logger.info(msg)
            return True, msg

        except Exception as e:
            msg = (
                f"Error moving {source_file.name} to {dest_directory}: "
                f"{str(e)}"
            )
            logger.error(msg)
            raise

    @staticmethod
    def copy_file(
        source_file: Path,
        dest_directory: Path,
        conflict_resolution: ConflictResolution = ConflictResolution.SKIP,
    ) -> Tuple[bool, str]:
        """Copy a file to a destination directory.

        Parameters
        ----------
        source_file : Path
            Path to the source file
        dest_directory : Path
            Path to destination directory
        conflict_resolution : ConflictResolution, optional
            How to handle existing files (default: SKIP)

        Returns
        -------
        success : bool
            True if operation succeeded, False otherwise
        message : str
            Status message describing what happened

        Raises
        ------
        FileNotFoundError
            If source file does not exist
        """
        source_file = Path(source_file)
        dest_directory = Path(dest_directory)

        if not source_file.exists():
            msg = f"Source file not found: {source_file}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Ensure destination directory exists
        FileOrganizer.ensure_directory_exists(dest_directory)

        dest_file = dest_directory / source_file.name

        # Handle file conflicts
        if dest_file.exists():
            if conflict_resolution == ConflictResolution.SKIP:
                msg = f"File already exists, skipping: {dest_file}"
                logger.info(msg)
                return False, msg

            elif conflict_resolution == ConflictResolution.OVERWRITE:
                dest_file.unlink()
                logger.debug(f"Overwriting existing file: {dest_file}")

            elif conflict_resolution == ConflictResolution.RENAME:
                dest_file = FileOrganizer._get_unique_filename(dest_file)
                logger.debug(f"Renamed to avoid conflict: {dest_file}")

        try:
            shutil.copy2(str(source_file), str(dest_file))
            msg = f"Copied {source_file.name} to {dest_directory}"
            logger.info(msg)
            return True, msg

        except Exception as e:
            msg = (
                f"Error copying {source_file.name} to {dest_directory}: "
                f"{str(e)}"
            )
            logger.error(msg)
            raise

    @staticmethod
    def _get_unique_filename(file_path: Path) -> Path:
        """Generate a unique filename by adding numeric suffix.

        Parameters
        ----------
        file_path : Path
            Original file path

        Returns
        -------
        unique_path : Path
            Modified path with numeric suffix before extension
            (e.g., "image.tiff" -> "image_1.tiff")
        """
        if not file_path.exists():
            return file_path

        stem = file_path.stem
        suffix = file_path.suffix
        parent = file_path.parent

        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    @staticmethod
    def organize_classification(
        source_file: Path,
        classification: str,
        output_base_directory: Path,
        operation: FileOperation = FileOperation.MOVE,
        conflict_resolution: ConflictResolution = ConflictResolution.RENAME,
    ) -> Tuple[bool, str]:
        """Organize a classified image into appropriate output directory.

        Moves or copies a file to a subdirectory named after the
        classification (e.g., "pits", "no_pits").

        Parameters
        ----------
        source_file : Path
            Path to the image file to organize
        classification : str
            Classification label (becomes subdirectory name)
        output_base_directory : Path
            Base directory where classification subdirectories are created
        operation : FileOperation, optional
            Whether to MOVE or COPY the file (default: MOVE)
        conflict_resolution : ConflictResolution, optional
            How to handle existing files (default: RENAME)

        Returns
        -------
        success : bool
            True if operation succeeded, False otherwise
        message : str
            Status message describing what happened

        Raises
        ------
        FileNotFoundError
            If source file does not exist
        ValueError
            If classification is invalid (empty or contains path separators)
        """
        source_file = Path(source_file)

        if not source_file.exists():
            msg = f"Source file not found: {source_file}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Validate classification
        classification = str(classification).strip()
        if not classification:
            msg = "Classification cannot be empty"
            logger.error(msg)
            raise ValueError(msg)

        if "/" in classification or "\\" in classification:
            msg = "Classification cannot contain path separators"
            logger.error(msg)
            raise ValueError(msg)

        # Create output directory
        output_directory = output_base_directory / classification
        FileOrganizer.ensure_directory_exists(output_directory)

        # Perform file operation
        if operation == FileOperation.MOVE:
            return FileOrganizer.move_file(
                source_file, output_directory, conflict_resolution
            )
        else:  # COPY
            return FileOrganizer.copy_file(
                source_file, output_directory, conflict_resolution
            )

    @staticmethod
    def create_classification_directories(
        output_base_directory: Path,
        classifications: list[str],
    ) -> dict:
        """Create all classification subdirectories.

        Parameters
        ----------
        output_base_directory : Path
            Base directory path
        classifications : list of str
            List of classification names

        Returns
        -------
        created_dirs : dict
            Dictionary mapping classification names to created directory paths

        Raises
        ------
        ValueError
            If classifications list is empty
        """
        if not classifications:
            msg = "Classifications list cannot be empty"
            logger.error(msg)
            raise ValueError(msg)

        created_dirs = {}

        for classification in classifications:
            classification = str(classification).strip()

            if not classification:
                logger.warning("Skipping empty classification")
                continue

            if "/" in classification or "\\" in classification:
                logger.warning(
                    f"Skipping classification with path separators: "
                    f"{classification}"
                )
                continue

            directory = output_base_directory / classification
            FileOrganizer.ensure_directory_exists(directory)
            created_dirs[classification] = directory

            logger.info(f"Created classification directory: {directory}")

        return created_dirs

    @staticmethod
    def get_classification_directories(
        output_base_directory: Path,
    ) -> dict:
        """Get all existing classification subdirectories.

        Parameters
        ----------
        output_base_directory : Path
            Base directory to search

        Returns
        -------
        classification_dirs : dict
            Dictionary mapping classification names to directory paths
        """
        output_base_directory = Path(output_base_directory)

        if not output_base_directory.exists():
            logger.warning(
                f"Base directory does not exist: {output_base_directory}"
            )
            return {}

        classification_dirs = {}

        for item in output_base_directory.iterdir():
            if item.is_dir():
                classification_dirs[item.name] = item

        return classification_dirs

    @staticmethod
    def count_files_in_classification(
        output_base_directory: Path,
        classification: str,
    ) -> int:
        """Count files in a classification directory.

        Parameters
        ----------
        output_base_directory : Path
            Base directory
        classification : str
            Classification name

        Returns
        -------
        count : int
            Number of files in the classification directory
        """
        classification_dir = output_base_directory / classification

        if not classification_dir.exists():
            return 0

        return len(list(classification_dir.glob("*")))

    @staticmethod
    def count_all_classifications(
        output_base_directory: Path,
    ) -> dict:
        """Count files in all classification directories.

        Parameters
        ----------
        output_base_directory : Path
            Base directory

        Returns
        -------
        counts : dict
            Dictionary mapping classification names to file counts
        """
        output_base_directory = Path(output_base_directory)

        if not output_base_directory.exists():
            logger.warning(
                f"Base directory does not exist: {output_base_directory}"
            )
            return {}

        counts = {}

        for item in output_base_directory.iterdir():
            if item.is_dir():
                count = len(list(item.glob("*")))
                counts[item.name] = count

        return counts
