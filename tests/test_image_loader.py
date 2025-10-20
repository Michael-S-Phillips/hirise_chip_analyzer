"""Tests for image_loader module.

This test suite validates TIFF image loading, validation, and metadata
extraction functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hirise_chip_analyzer.core.image_loader import ImageLoader


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample grayscale image array."""
    return np.random.randint(0, 256, (256, 256), dtype=np.uint8)


@pytest.fixture
def sample_color_image():
    """Create a sample color image array."""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_tiff_file(temp_image_dir, sample_image):
    """Create a sample TIFF file in temporary directory."""
    file_path = temp_image_dir / "test_image.tiff"
    img = Image.fromarray(sample_image)
    img.save(file_path)
    return file_path


@pytest.fixture
def multiple_tiff_files(temp_image_dir):
    """Create multiple TIFF files in temporary directory."""
    files = []
    for i in range(3):
        file_path = temp_image_dir / f"image_{i}.tiff"
        img_array = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(file_path)
        files.append(file_path)
    return files


class TestImageLoaderBasic:
    """Test basic image loading functionality."""

    def test_load_single_grayscale_image(self, sample_tiff_file):
        """Test loading a single grayscale TIFF image."""
        image_data, metadata = ImageLoader.load_image(sample_tiff_file)

        assert isinstance(image_data, np.ndarray)
        assert image_data.shape == (256, 256)
        assert image_data.dtype == np.uint8
        assert metadata['filename'] == 'test_image.tiff'
        assert metadata['shape'] == (256, 256)
        assert metadata['bands'] == 1

    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises FileNotFoundError."""
        fake_path = Path("/nonexistent/path/image.tiff")

        with pytest.raises(FileNotFoundError):
            ImageLoader.load_image(fake_path)

    def test_load_invalid_file_format(self, temp_image_dir):
        """Test that loading an invalid file raises ValueError."""
        invalid_file = temp_image_dir / "not_an_image.txt"
        invalid_file.write_text("This is not an image")

        with pytest.raises(ValueError):
            ImageLoader.load_image(invalid_file)

    def test_metadata_contains_required_fields(self, sample_tiff_file):
        """Test that metadata contains all required fields."""
        _, metadata = ImageLoader.load_image(sample_tiff_file)

        required_fields = [
            'filename', 'shape', 'dtype', 'bands', 'size_bytes'
        ]
        for field in required_fields:
            assert field in metadata

    def test_size_bytes_metadata(self, sample_tiff_file):
        """Test that size_bytes metadata is correct."""
        _, metadata = ImageLoader.load_image(sample_tiff_file)

        actual_size = sample_tiff_file.stat().st_size
        assert metadata['size_bytes'] == actual_size
        assert metadata['size_bytes'] > 0


class TestDirectoryLoading:
    """Test loading images from directories."""

    def test_load_images_from_directory(self, multiple_tiff_files, temp_image_dir):
        """Test loading multiple images from a directory."""
        images = ImageLoader.load_images_from_directory(temp_image_dir)

        assert len(images) == 3
        for image_data, metadata in images:
            assert isinstance(image_data, np.ndarray)
            assert image_data.shape == (128, 128)
            assert 'filename' in metadata

    def test_load_from_nonexistent_directory(self):
        """Test that loading from nonexistent directory raises error."""
        fake_dir = Path("/nonexistent/directory")

        with pytest.raises(FileNotFoundError):
            ImageLoader.load_images_from_directory(fake_dir)

    def test_load_from_empty_directory(self, temp_image_dir):
        """Test loading from empty directory returns empty list."""
        images = ImageLoader.load_images_from_directory(temp_image_dir)
        assert images == []

    def test_get_image_filenames(self, multiple_tiff_files, temp_image_dir):
        """Test getting list of image filenames from directory."""
        filenames = ImageLoader.get_image_filenames(temp_image_dir)

        assert len(filenames) == 3
        assert all(isinstance(f, Path) for f in filenames)
        assert all(f.suffix.lower() in ['.tiff', '.tif'] for f in filenames)

    def test_get_image_filenames_empty_directory(self, temp_image_dir):
        """Test getting filenames from empty directory."""
        filenames = ImageLoader.get_image_filenames(temp_image_dir)
        assert filenames == []

    def test_filenames_are_sorted(self, multiple_tiff_files, temp_image_dir):
        """Test that returned filenames are sorted."""
        filenames = ImageLoader.get_image_filenames(temp_image_dir)
        assert filenames == sorted(filenames)


class TestImageValidation:
    """Test image validation functionality."""

    def test_validate_valid_image(self, sample_image):
        """Test validation of a valid image."""
        assert ImageLoader.validate_image(sample_image) is True

    def test_validate_with_expected_shape(self, sample_image):
        """Test validation with expected shape."""
        assert ImageLoader.validate_image(
            sample_image, expected_shape=(256, 256)
        ) is True

    def test_validate_wrong_expected_shape(self, sample_image):
        """Test that wrong expected shape raises error."""
        with pytest.raises(ValueError):
            ImageLoader.validate_image(
                sample_image, expected_shape=(512, 512)
            )

    def test_validate_not_ndarray(self):
        """Test that non-ndarray input raises error."""
        with pytest.raises(ValueError):
            ImageLoader.validate_image([1, 2, 3])

    def test_validate_1d_array(self):
        """Test that 1D array raises error."""
        array_1d = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            ImageLoader.validate_image(array_1d)

    def test_validate_with_nan_values(self):
        """Test validation with NaN values."""
        image_with_nan = np.full((100, 100), np.nan, dtype=np.float32)
        # Should not raise, just log warning
        result = ImageLoader.validate_image(image_with_nan)
        assert result is True

    def test_validate_with_inf_values(self):
        """Test validation with infinite values."""
        image_with_inf = np.full((100, 100), np.inf, dtype=np.float32)
        # Should not raise, just log warning
        result = ImageLoader.validate_image(image_with_inf)
        assert result is True


class TestImageStatistics:
    """Test image statistics calculation."""

    def test_get_stats_basic(self):
        """Test basic statistical calculation."""
        image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        stats = ImageLoader.get_image_stats(image)

        assert stats['min'] == 10
        assert stats['max'] == 40
        assert stats['mean'] == 25.0
        assert 'std' in stats
        assert 'median' in stats

    def test_stats_contains_required_fields(self, sample_image):
        """Test that stats contain all required fields."""
        stats = ImageLoader.get_image_stats(sample_image)

        required_fields = [
            'min', 'max', 'mean', 'std', 'median', 'percentiles'
        ]
        for field in required_fields:
            assert field in stats

    def test_percentiles_in_stats(self, sample_image):
        """Test that percentiles are correctly calculated."""
        stats = ImageLoader.get_image_stats(sample_image)

        assert '1' in stats['percentiles']
        assert '5' in stats['percentiles']
        assert '95' in stats['percentiles']
        assert '99' in stats['percentiles']

    def test_percentiles_ordering(self, sample_image):
        """Test that percentiles are in correct order."""
        stats = ImageLoader.get_image_stats(sample_image)

        p1 = stats['percentiles']['1']
        p5 = stats['percentiles']['5']
        p95 = stats['percentiles']['95']
        p99 = stats['percentiles']['99']

        assert p1 <= p5 <= p95 <= p99

    def test_stats_with_constant_image(self):
        """Test statistics with constant valued image."""
        image = np.full((100, 100), 50, dtype=np.uint8)
        stats = ImageLoader.get_image_stats(image)

        assert stats['min'] == 50
        assert stats['max'] == 50
        assert stats['mean'] == 50
        assert stats['std'] == 0
        assert stats['median'] == 50


class TestFileFormatSupport:
    """Test support for different TIFF file extensions."""

    def test_load_tiff_extension(self, temp_image_dir):
        """Test loading .tiff extension."""
        file_path = temp_image_dir / "image.tiff"
        img_array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        Image.fromarray(img_array).save(file_path)

        image_data, metadata = ImageLoader.load_image(file_path)
        assert image_data.shape == (64, 64)

    def test_load_tif_extension(self, temp_image_dir):
        """Test loading .tif extension."""
        file_path = temp_image_dir / "image.tif"
        img_array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        Image.fromarray(img_array).save(file_path)

        image_data, metadata = ImageLoader.load_image(file_path)
        assert image_data.shape == (64, 64)

    def test_get_filenames_finds_both_extensions(self, temp_image_dir):
        """Test that get_image_filenames finds both .tiff and .tif."""
        (temp_image_dir / "image1.tiff").write_bytes(b'')
        (temp_image_dir / "image2.tif").write_bytes(b'')

        # Manually create valid TIFF files
        for filename in ["image1.tiff", "image2.tif"]:
            img_array = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            Image.fromarray(img_array).save(temp_image_dir / filename)

        filenames = ImageLoader.get_image_filenames(temp_image_dir)
        assert len(filenames) == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_load_small_image(self, temp_image_dir):
        """Test loading very small image."""
        file_path = temp_image_dir / "tiny.tiff"
        img_array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        Image.fromarray(img_array).save(file_path)

        image_data, _ = ImageLoader.load_image(file_path)
        assert image_data.shape == (2, 2)

    def test_stats_with_zeros(self):
        """Test statistics with all-zero image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        stats = ImageLoader.get_image_stats(image)

        assert stats['min'] == 0
        assert stats['max'] == 0
        assert stats['mean'] == 0
        assert stats['std'] == 0

    def test_stats_with_full_range(self):
        """Test statistics with full 0-255 range."""
        image = np.linspace(0, 255, 256*256).reshape(256, 256).astype(
            np.uint8
        )
        stats = ImageLoader.get_image_stats(image)

        assert stats['min'] >= 0
        assert stats['max'] <= 255
