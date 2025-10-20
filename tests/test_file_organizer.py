"""Tests for file_organizer module.

This test suite validates file organization, movement, copying, and
classification directory management functionality.
"""

import tempfile
from pathlib import Path

import pytest

from hirise_chip_analyzer.core.file_organizer import (
    ConflictResolution,
    FileOperation,
    FileOrganizer,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_file(temp_dir):
    """Create a test file."""
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("Test content")
    return file_path


@pytest.fixture
def test_file_copy(temp_dir, test_file):
    """Create a copy of test file."""
    copy_path = temp_dir / "test_file_copy.txt"
    copy_path.write_text(test_file.read_text())
    return copy_path


class TestDirectoryEnsure:
    """Test directory creation functionality."""

    def test_ensure_existing_directory(self, temp_dir):
        """Test that existing directory is returned unchanged."""
        result = FileOrganizer.ensure_directory_exists(temp_dir)
        assert result == temp_dir
        assert temp_dir.exists()

    def test_ensure_creates_new_directory(self, temp_dir):
        """Test that new directory is created."""
        new_dir = temp_dir / "new_directory"
        assert not new_dir.exists()

        result = FileOrganizer.ensure_directory_exists(new_dir)

        assert result == new_dir
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_creates_nested_directories(self, temp_dir):
        """Test that nested directories are created."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        result = FileOrganizer.ensure_directory_exists(nested_dir)

        assert result == nested_dir
        assert nested_dir.exists()

    def test_ensure_permission_error(self, monkeypatch):
        """Test handling of permission error."""
        def mock_mkdir(self, **kwargs):
            raise PermissionError("Permission denied")

        monkeypatch.setattr(Path, "mkdir", mock_mkdir)

        with pytest.raises(PermissionError):
            FileOrganizer.ensure_directory_exists(Path("/test"))


class TestMoveFile:
    """Test file moving functionality."""

    def test_move_file_basic(self, temp_dir, test_file):
        """Test moving a file to another directory."""
        dest_dir = temp_dir / "destination"
        dest_dir.mkdir()

        success, message = FileOrganizer.move_file(test_file, dest_dir)

        assert success is True
        assert not test_file.exists()
        assert (dest_dir / "test_file.txt").exists()

    def test_move_file_creates_destination(self, temp_dir, test_file):
        """Test that destination directory is created if needed."""
        dest_dir = temp_dir / "new_dest"
        assert not dest_dir.exists()

        success, message = FileOrganizer.move_file(test_file, dest_dir)

        assert success is True
        assert dest_dir.exists()
        assert (dest_dir / "test_file.txt").exists()

    def test_move_nonexistent_file(self, temp_dir):
        """Test that moving nonexistent file raises error."""
        fake_file = temp_dir / "nonexistent.txt"
        dest_dir = temp_dir / "destination"

        with pytest.raises(FileNotFoundError):
            FileOrganizer.move_file(fake_file, dest_dir)

    def test_move_file_skip_on_conflict(self, temp_dir, test_file,
                                        test_file_copy):
        """Test SKIP conflict resolution."""
        dest_dir = temp_dir / "destination"
        dest_dir.mkdir()

        # Copy existing file to destination
        existing = dest_dir / "test_file.txt"
        existing.write_text("Existing content")

        success, message = FileOrganizer.move_file(
            test_file, dest_dir,
            conflict_resolution=ConflictResolution.SKIP
        )

        assert success is False
        assert test_file.exists()  # Should not be moved
        assert existing.read_text() == "Existing content"

    def test_move_file_overwrite_on_conflict(self, temp_dir, test_file):
        """Test OVERWRITE conflict resolution."""
        dest_dir = temp_dir / "destination"
        dest_dir.mkdir()

        # Create existing file
        existing = dest_dir / "test_file.txt"
        existing.write_text("Old content")

        success, message = FileOrganizer.move_file(
            test_file, dest_dir,
            conflict_resolution=ConflictResolution.OVERWRITE
        )

        assert success is True
        assert not test_file.exists()
        assert existing.exists()
        assert existing.read_text() == "Test content"

    def test_move_file_rename_on_conflict(self, temp_dir, test_file):
        """Test RENAME conflict resolution."""
        dest_dir = temp_dir / "destination"
        dest_dir.mkdir()

        # Create existing file
        existing = dest_dir / "test_file.txt"
        existing.write_text("Existing content")

        success, message = FileOrganizer.move_file(
            test_file, dest_dir,
            conflict_resolution=ConflictResolution.RENAME
        )

        assert success is True
        assert not test_file.exists()
        assert existing.exists()
        assert (dest_dir / "test_file_1.txt").exists()


class TestCopyFile:
    """Test file copying functionality."""

    def test_copy_file_basic(self, temp_dir, test_file):
        """Test copying a file to another directory."""
        dest_dir = temp_dir / "destination"
        dest_dir.mkdir()

        success, message = FileOrganizer.copy_file(test_file, dest_dir)

        assert success is True
        assert test_file.exists()  # Original still exists
        assert (dest_dir / "test_file.txt").exists()
        assert (dest_dir / "test_file.txt").read_text() == "Test content"

    def test_copy_file_creates_destination(self, temp_dir, test_file):
        """Test that destination directory is created if needed."""
        dest_dir = temp_dir / "new_dest"
        assert not dest_dir.exists()

        success, message = FileOrganizer.copy_file(test_file, dest_dir)

        assert success is True
        assert dest_dir.exists()
        assert (dest_dir / "test_file.txt").exists()

    def test_copy_nonexistent_file(self, temp_dir):
        """Test that copying nonexistent file raises error."""
        fake_file = temp_dir / "nonexistent.txt"
        dest_dir = temp_dir / "destination"

        with pytest.raises(FileNotFoundError):
            FileOrganizer.copy_file(fake_file, dest_dir)

    def test_copy_file_skip_on_conflict(self, temp_dir, test_file):
        """Test SKIP conflict resolution for copy."""
        dest_dir = temp_dir / "destination"
        dest_dir.mkdir()

        # Create existing file
        existing = dest_dir / "test_file.txt"
        existing.write_text("Existing content")

        success, message = FileOrganizer.copy_file(
            test_file, dest_dir,
            conflict_resolution=ConflictResolution.SKIP
        )

        assert success is False
        assert test_file.exists()
        assert existing.read_text() == "Existing content"

    def test_copy_file_rename_on_conflict(self, temp_dir, test_file):
        """Test RENAME conflict resolution for copy."""
        dest_dir = temp_dir / "destination"
        dest_dir.mkdir()

        # Create existing file
        existing = dest_dir / "test_file.txt"
        existing.write_text("Existing content")

        success, message = FileOrganizer.copy_file(
            test_file, dest_dir,
            conflict_resolution=ConflictResolution.RENAME
        )

        assert success is True
        assert test_file.exists()
        assert existing.exists()
        assert (dest_dir / "test_file_1.txt").exists()


class TestOrganizeClassification:
    """Test classification-based organization."""

    def test_organize_with_move(self, temp_dir, test_file):
        """Test organizing file with MOVE operation."""
        output_base = temp_dir / "output"

        success, message = FileOrganizer.organize_classification(
            test_file,
            "pits",
            output_base,
            operation=FileOperation.MOVE
        )

        assert success is True
        assert not test_file.exists()
        assert (output_base / "pits" / "test_file.txt").exists()

    def test_organize_with_copy(self, temp_dir, test_file):
        """Test organizing file with COPY operation."""
        output_base = temp_dir / "output"

        success, message = FileOrganizer.organize_classification(
            test_file,
            "no_pits",
            output_base,
            operation=FileOperation.COPY
        )

        assert success is True
        assert test_file.exists()  # Original still exists
        assert (output_base / "no_pits" / "test_file.txt").exists()

    def test_organize_creates_classification_directory(self, temp_dir,
                                                       test_file):
        """Test that classification directory is created."""
        output_base = temp_dir / "output"

        FileOrganizer.organize_classification(
            test_file,
            "uncertain",
            output_base
        )

        assert (output_base / "uncertain").exists()
        assert (output_base / "uncertain").is_dir()

    def test_organize_invalid_classification_empty(self, temp_dir,
                                                    test_file):
        """Test that empty classification raises error."""
        output_base = temp_dir / "output"

        with pytest.raises(ValueError):
            FileOrganizer.organize_classification(
                test_file,
                "",
                output_base
            )

    def test_organize_invalid_classification_path_separators(
        self, temp_dir, test_file
    ):
        """Test that classification with path separators raises error."""
        output_base = temp_dir / "output"

        with pytest.raises(ValueError):
            FileOrganizer.organize_classification(
                test_file,
                "pits/subcategory",
                output_base
            )

    def test_organize_nonexistent_file(self, temp_dir):
        """Test that organizing nonexistent file raises error."""
        output_base = temp_dir / "output"

        with pytest.raises(FileNotFoundError):
            FileOrganizer.organize_classification(
                Path("nonexistent.txt"),
                "pits",
                output_base
            )


class TestCreateClassificationDirectories:
    """Test batch directory creation."""

    def test_create_single_classification(self, temp_dir):
        """Test creating single classification directory."""
        output_base = temp_dir / "output"

        dirs = FileOrganizer.create_classification_directories(
            output_base,
            ["pits"]
        )

        assert "pits" in dirs
        assert (output_base / "pits").exists()

    def test_create_multiple_classifications(self, temp_dir):
        """Test creating multiple classification directories."""
        output_base = temp_dir / "output"
        classifications = ["pits", "no_pits", "uncertain"]

        dirs = FileOrganizer.create_classification_directories(
            output_base,
            classifications
        )

        assert len(dirs) == 3
        for classification in classifications:
            assert (output_base / classification).exists()

    def test_create_empty_classifications_raises_error(self, temp_dir):
        """Test that empty classifications list raises error."""
        output_base = temp_dir / "output"

        with pytest.raises(ValueError):
            FileOrganizer.create_classification_directories(
                output_base,
                []
            )

    def test_create_skips_invalid_classifications(self, temp_dir):
        """Test that invalid classifications are skipped."""
        output_base = temp_dir / "output"

        dirs = FileOrganizer.create_classification_directories(
            output_base,
            ["valid", "", "invalid/path", "another_valid"]
        )

        assert "valid" in dirs
        assert "" not in dirs
        assert "invalid/path" not in dirs
        assert "another_valid" in dirs


class TestGetClassificationDirectories:
    """Test retrieving classification directories."""

    def test_get_classification_directories(self, temp_dir):
        """Test getting existing classification directories."""
        output_base = temp_dir / "output"
        (output_base / "pits").mkdir(parents=True)
        (output_base / "no_pits").mkdir(parents=True)
        (output_base / "file.txt").write_text("not a dir")

        dirs = FileOrganizer.get_classification_directories(output_base)

        assert "pits" in dirs
        assert "no_pits" in dirs
        assert len(dirs) == 2

    def test_get_from_nonexistent_directory(self, temp_dir):
        """Test getting from nonexistent directory returns empty dict."""
        nonexistent = temp_dir / "nonexistent"

        dirs = FileOrganizer.get_classification_directories(nonexistent)

        assert dirs == {}


class TestCountFiles:
    """Test file counting functionality."""

    def test_count_files_in_classification(self, temp_dir):
        """Test counting files in a classification directory."""
        output_base = temp_dir / "output"
        class_dir = output_base / "pits"
        class_dir.mkdir(parents=True)

        # Create test files
        for i in range(3):
            (class_dir / f"image_{i}.tif").write_text("data")

        count = FileOrganizer.count_files_in_classification(
            output_base,
            "pits"
        )

        assert count == 3

    def test_count_files_nonexistent_classification(self, temp_dir):
        """Test counting files in nonexistent classification."""
        output_base = temp_dir / "output"

        count = FileOrganizer.count_files_in_classification(
            output_base,
            "nonexistent"
        )

        assert count == 0

    def test_count_all_classifications(self, temp_dir):
        """Test counting files in all classifications."""
        output_base = temp_dir / "output"

        # Create directories with different file counts
        (output_base / "pits").mkdir(parents=True)
        (output_base / "no_pits").mkdir(parents=True)

        for i in range(3):
            (output_base / "pits" / f"img_{i}.tif").write_text("data")
        for i in range(2):
            (output_base / "no_pits" / f"img_{i}.tif").write_text("data")

        counts = FileOrganizer.count_all_classifications(output_base)

        assert counts["pits"] == 3
        assert counts["no_pits"] == 2


class TestUniqueFilename:
    """Test unique filename generation."""

    def test_get_unique_filename_nonexistent(self, temp_dir):
        """Test unique filename for nonexistent file."""
        file_path = temp_dir / "image.tif"

        unique_path = FileOrganizer._get_unique_filename(file_path)

        assert unique_path == file_path

    def test_get_unique_filename_existing(self, temp_dir):
        """Test unique filename for existing file."""
        file_path = temp_dir / "image.tif"
        file_path.write_text("data")

        unique_path = FileOrganizer._get_unique_filename(file_path)

        assert unique_path.name == "image_1.tif"
        assert unique_path != file_path

    def test_get_unique_filename_multiple_conflicts(self, temp_dir):
        """Test unique filename with multiple existing files."""
        file_path = temp_dir / "image.tif"
        file_path.write_text("data")
        (temp_dir / "image_1.tif").write_text("data")
        (temp_dir / "image_2.tif").write_text("data")

        unique_path = FileOrganizer._get_unique_filename(file_path)

        assert unique_path.name == "image_3.tif"
