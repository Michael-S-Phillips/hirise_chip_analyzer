"""Tests for classifier module.

This test suite validates classification tracking, statistics, and
session management functionality.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from hirise_chip_analyzer.core.classifier import (
    ClassificationRecord,
    Classifier,
)


class TestClassificationRecord:
    """Test ClassificationRecord class."""

    def test_create_record(self):
        """Test creating a classification record."""
        record = ClassificationRecord("image.tiff", "pits")

        assert record.filename == "image.tiff"
        assert record.classification == "pits"
        assert isinstance(record.timestamp, datetime)

    def test_create_record_with_notes(self):
        """Test creating a record with notes."""
        record = ClassificationRecord(
            "image.tiff", "pits",
            notes="Large pit formation"
        )

        assert record.notes == "Large pit formation"

    def test_create_record_with_custom_timestamp(self):
        """Test creating a record with custom timestamp."""
        now = datetime.now()
        record = ClassificationRecord(
            "image.tiff", "pits",
            timestamp=now
        )

        assert record.timestamp == now

    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = ClassificationRecord("image.tiff", "pits")
        record_dict = record.to_dict()

        assert record_dict['filename'] == "image.tiff"
        assert record_dict['classification'] == "pits"
        assert 'timestamp' in record_dict
        assert record_dict['notes'] is None

    def test_record_from_dict(self):
        """Test creating record from dictionary."""
        now = datetime.now()
        data = {
            'filename': 'image.tiff',
            'classification': 'no_pits',
            'timestamp': now.isoformat(),
            'notes': 'No visible pits',
        }

        record = ClassificationRecord.from_dict(data)

        assert record.filename == "image.tiff"
        assert record.classification == "no_pits"
        assert record.notes == "No visible pits"

    def test_record_round_trip(self):
        """Test record serialization and deserialization."""
        original = ClassificationRecord(
            "image.tiff", "pits",
            notes="Test notes"
        )

        record_dict = original.to_dict()
        restored = ClassificationRecord.from_dict(record_dict)

        assert restored.filename == original.filename
        assert restored.classification == original.classification
        assert restored.notes == original.notes


class TestClassifierBasic:
    """Test basic classifier functionality."""

    def test_create_classifier(self):
        """Test creating a new classifier."""
        classifier = Classifier()

        assert classifier.get_total_classified() == 0
        assert isinstance(classifier.session_start, datetime)

    def test_classify_image(self):
        """Test classifying an image."""
        classifier = Classifier()

        success = classifier.classify_image("image.tiff", "pits")

        assert success is True
        assert classifier.get_total_classified() == 1

    def test_classify_multiple_images(self):
        """Test classifying multiple images."""
        classifier = Classifier()

        classifier.classify_image("image1.tiff", "pits")
        classifier.classify_image("image2.tiff", "no_pits")
        classifier.classify_image("image3.tiff", "pits")

        assert classifier.get_total_classified() == 3

    def test_classify_with_notes(self):
        """Test classification with notes."""
        classifier = Classifier()

        classifier.classify_image(
            "image.tiff", "pits",
            notes="Large formation"
        )

        records = classifier.get_records()
        assert records[0].notes == "Large formation"

    def test_classify_invalid_filename(self):
        """Test that empty filename raises error."""
        classifier = Classifier()

        with pytest.raises(ValueError):
            classifier.classify_image("", "pits")

    def test_classify_invalid_classification(self):
        """Test that empty classification raises error."""
        classifier = Classifier()

        with pytest.raises(ValueError):
            classifier.classify_image("image.tiff", "")


class TestClassifierStatistics:
    """Test classifier statistics functionality."""

    def test_get_classification_count(self):
        """Test getting count for specific classification."""
        classifier = Classifier()

        classifier.classify_image("image1.tiff", "pits")
        classifier.classify_image("image2.tiff", "pits")
        classifier.classify_image("image3.tiff", "no_pits")

        assert classifier.get_classification_count("pits") == 2
        assert classifier.get_classification_count("no_pits") == 1

    def test_get_all_classifications(self):
        """Test getting all classification counts."""
        classifier = Classifier()

        classifier.classify_image("image1.tiff", "pits")
        classifier.classify_image("image2.tiff", "no_pits")
        classifier.classify_image("image3.tiff", "uncertain")
        classifier.classify_image("image4.tiff", "pits")

        classifications = classifier.get_all_classifications()

        assert classifications["pits"] == 2
        assert classifications["no_pits"] == 1
        assert classifications["uncertain"] == 1

    def test_get_statistics(self):
        """Test getting comprehensive statistics."""
        classifier = Classifier()

        classifier.classify_image("image1.tiff", "pits")
        classifier.classify_image("image2.tiff", "no_pits")

        stats = classifier.get_statistics()

        assert stats['total_classified'] == 2
        assert 'classifications' in stats
        assert 'session_duration' in stats
        assert 'start_time' in stats

    def test_statistics_structure(self):
        """Test that statistics has all required fields."""
        classifier = Classifier()
        classifier.classify_image("image.tiff", "pits")

        stats = classifier.get_statistics()

        required_fields = [
            'total_classified', 'classifications',
            'session_duration', 'start_time'
        ]
        for field in required_fields:
            assert field in stats

    def test_get_records(self):
        """Test getting all records."""
        classifier = Classifier()

        classifier.classify_image("image1.tiff", "pits")
        classifier.classify_image("image2.tiff", "no_pits")

        records = classifier.get_records()

        assert len(records) == 2
        assert isinstance(records[0], ClassificationRecord)

    def test_get_records_by_classification(self):
        """Test getting records filtered by classification."""
        classifier = Classifier()

        classifier.classify_image("image1.tiff", "pits")
        classifier.classify_image("image2.tiff", "no_pits")
        classifier.classify_image("image3.tiff", "pits")

        pit_records = classifier.get_records_by_classification("pits")
        no_pit_records = classifier.get_records_by_classification("no_pits")

        assert len(pit_records) == 2
        assert len(no_pit_records) == 1


class TestClassifierHistory:
    """Test classifier history management."""

    def test_clear_history(self):
        """Test clearing classification history."""
        classifier = Classifier()

        classifier.classify_image("image1.tiff", "pits")
        classifier.classify_image("image2.tiff", "no_pits")

        count = classifier.clear_history()

        assert count == 2
        assert classifier.get_total_classified() == 0

    def test_get_records_returns_copy(self):
        """Test that get_records returns a copy."""
        classifier = Classifier()

        classifier.classify_image("image1.tiff", "pits")

        records = classifier.get_records()
        original_count = len(records)

        classifier.classify_image("image2.tiff", "no_pits")

        # Original returned list should not change
        assert len(records) == original_count


class TestClassifierSerialization:
    """Test classifier session saving and loading."""

    def test_save_session(self):
        """Test saving classifier session to file."""
        classifier = Classifier()
        classifier.classify_image("image1.tiff", "pits")
        classifier.classify_image("image2.tiff", "no_pits")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "session.json"
            success = classifier.save_session(output_file)

            assert success is True
            assert output_file.exists()

    def test_save_session_creates_valid_json(self):
        """Test that saved session is valid JSON."""
        classifier = Classifier()
        classifier.classify_image("image1.tiff", "pits")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "session.json"
            classifier.save_session(output_file)

            with open(output_file, 'r') as f:
                data = json.load(f)

            assert 'session_start' in data
            assert 'records' in data
            assert 'statistics' in data
            assert len(data['records']) == 1

    def test_load_session(self):
        """Test loading classifier session from file."""
        # Create and save a session
        classifier1 = Classifier()
        classifier1.classify_image("image1.tiff", "pits")
        classifier1.classify_image("image2.tiff", "no_pits")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "session.json"
            classifier1.save_session(output_file)

            # Load the session
            classifier2 = Classifier.load_session(output_file)

            assert classifier2.get_total_classified() == 2
            assert classifier2.get_classification_count("pits") == 1
            assert classifier2.get_classification_count("no_pits") == 1

    def test_load_nonexistent_session(self):
        """Test that loading nonexistent session raises error."""
        fake_file = Path("/nonexistent/session.json")

        with pytest.raises(FileNotFoundError):
            Classifier.load_session(fake_file)

    def test_load_invalid_json_session(self):
        """Test that loading invalid JSON raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "invalid.json"
            output_file.write_text("{ invalid json")

            with pytest.raises(ValueError):
                Classifier.load_session(output_file)

    def test_session_round_trip(self):
        """Test saving and loading a session preserves data."""
        original = Classifier()
        original.classify_image("image1.tiff", "pits", notes="Test note")
        original.classify_image("image2.tiff", "no_pits")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "session.json"
            original.save_session(output_file)

            loaded = Classifier.load_session(output_file)

            assert loaded.get_total_classified() == original.get_total_classified()
            assert loaded.get_all_classifications() == original.get_all_classifications()

            original_records = original.get_records()
            loaded_records = loaded.get_records()

            for orig, load in zip(original_records, loaded_records):
                assert orig.filename == load.filename
                assert orig.classification == load.classification
                assert orig.notes == load.notes


class TestClassifierExport:
    """Test classifier export functionality."""

    def test_export_summary(self):
        """Test exporting classification summary."""
        classifier = Classifier()
        classifier.classify_image("image1.tiff", "pits")
        classifier.classify_image("image2.tiff", "pits")
        classifier.classify_image("image3.tiff", "no_pits")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "summary.txt"
            success = classifier.export_summary(output_file)

            assert success is True
            assert output_file.exists()

    def test_export_summary_content(self):
        """Test that exported summary contains expected content."""
        classifier = Classifier()
        classifier.classify_image("image1.tiff", "pits")
        classifier.classify_image("image2.tiff", "no_pits")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "summary.txt"
            classifier.export_summary(output_file)

            content = output_file.read_text()

            assert "Classification Summary" in content
            assert "Total Classified: 2" in content
            assert "pits" in content
            assert "no_pits" in content

    def test_export_summary_percentages(self):
        """Test that export includes percentages."""
        classifier = Classifier()
        classifier.classify_image("image1.tiff", "pits")
        classifier.classify_image("image2.tiff", "pits")
        classifier.classify_image("image3.tiff", "no_pits")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "summary.txt"
            classifier.export_summary(output_file)

            content = output_file.read_text()

            # Should have percentages
            assert "%" in content


class TestClassifierEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_classify_same_image_twice(self):
        """Test classifying the same image filename twice."""
        classifier = Classifier()

        classifier.classify_image("image.tiff", "pits")
        classifier.classify_image("image.tiff", "no_pits")

        # Both should be recorded (even though it's the same filename)
        assert classifier.get_total_classified() == 2

    def test_statistics_with_no_classifications(self):
        """Test getting statistics with no classifications."""
        classifier = Classifier()

        stats = classifier.get_statistics()

        assert stats['total_classified'] == 0
        assert stats['classifications'] == {}

    def test_large_number_of_classifications(self):
        """Test handling large number of classifications."""
        classifier = Classifier()

        for i in range(1000):
            classification = "pits" if i % 2 == 0 else "no_pits"
            classifier.classify_image(f"image_{i}.tiff", classification)

        assert classifier.get_total_classified() == 1000
        assert classifier.get_classification_count("pits") == 500
        assert classifier.get_classification_count("no_pits") == 500

    def test_special_characters_in_filename(self):
        """Test classification with special characters in filename."""
        classifier = Classifier()

        classifier.classify_image("image_@#$%.tiff", "pits")
        classifier.classify_image("image (1).tiff", "no_pits")

        assert classifier.get_total_classified() == 2

    def test_unicode_classification(self):
        """Test classification with unicode characters."""
        classifier = Classifier()

        classifier.classify_image("image.tiff", "坑")  # Chinese character

        assert classifier.get_total_classified() == 1
        assert classifier.get_classification_count("坑") == 1
