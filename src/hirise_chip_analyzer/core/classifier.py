"""Classification management and tracking.

This module provides functionality to track image classifications,
maintain classification history, and manage classification sessions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ClassificationRecord:
    """Record of a single image classification.

    Attributes
    ----------
    filename : str
        Name of the classified image file
    classification : str
        Classification label (e.g., "pits", "no_pits")
    timestamp : datetime
        When the classification was made
    notes : str, optional
        Optional notes about the classification
    """

    def __init__(
        self,
        filename: str,
        classification: str,
        timestamp: Optional[datetime] = None,
        notes: Optional[str] = None,
    ):
        """Initialize a classification record.

        Parameters
        ----------
        filename : str
            Name of the classified image
        classification : str
            Classification label
        timestamp : datetime, optional
            Timestamp of classification (default: now)
        notes : str, optional
            Optional notes (default: None)
        """
        self.filename = filename
        self.classification = classification
        self.timestamp = timestamp or datetime.now()
        self.notes = notes

    def to_dict(self) -> dict:
        """Convert record to dictionary.

        Returns
        -------
        record_dict : dict
            Dictionary representation of the record
        """
        return {
            'filename': self.filename,
            'classification': self.classification,
            'timestamp': self.timestamp.isoformat(),
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ClassificationRecord':
        """Create record from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with record data

        Returns
        -------
        record : ClassificationRecord
            Reconstructed classification record
        """
        timestamp = datetime.fromisoformat(data['timestamp'])
        return cls(
            filename=data['filename'],
            classification=data['classification'],
            timestamp=timestamp,
            notes=data.get('notes'),
        )


class Classifier:
    """Manage image classifications and track classification history.

    This class maintains a session of image classifications, provides
    statistics, and can save/load classification history.
    """

    def __init__(self):
        """Initialize a new classifier session."""
        self.records: List[ClassificationRecord] = []
        self.session_start = datetime.now()
        logger.info("New classification session started")

    def classify_image(
        self,
        filename: str,
        classification: str,
        notes: Optional[str] = None,
    ) -> bool:
        """Record a classification for an image.

        Parameters
        ----------
        filename : str
            Name of the image file
        classification : str
            Classification label
        notes : str, optional
            Optional notes about the classification

        Returns
        -------
        success : bool
            True if classification was recorded successfully

        Raises
        ------
        ValueError
            If classification or filename is invalid
        """
        if not filename or not isinstance(filename, str):
            msg = "Filename must be a non-empty string"
            logger.error(msg)
            raise ValueError(msg)

        if not classification or not isinstance(classification, str):
            msg = "Classification must be a non-empty string"
            logger.error(msg)
            raise ValueError(msg)

        record = ClassificationRecord(
            filename=filename,
            classification=classification,
            notes=notes,
        )
        self.records.append(record)

        logger.info(
            f"Classified {filename} as '{classification}'"
        )
        return True

    def get_classification_count(self, classification: str) -> int:
        """Get number of images with a specific classification.

        Parameters
        ----------
        classification : str
            Classification label to count

        Returns
        -------
        count : int
            Number of images with this classification
        """
        return sum(
            1 for record in self.records
            if record.classification == classification
        )

    def get_all_classifications(self) -> dict:
        """Get count of all classifications.

        Returns
        -------
        classifications : dict
            Dictionary mapping classification labels to counts
        """
        classifications = {}
        for record in self.records:
            classification = record.classification
            classifications[classification] = (
                classifications.get(classification, 0) + 1
            )
        return classifications

    def get_total_classified(self) -> int:
        """Get total number of classified images.

        Returns
        -------
        total : int
            Total number of classifications recorded
        """
        return len(self.records)

    def get_records(self) -> List[ClassificationRecord]:
        """Get all classification records.

        Returns
        -------
        records : list of ClassificationRecord
            All classification records in order
        """
        return self.records.copy()

    def get_records_by_classification(
        self, classification: str
    ) -> List[ClassificationRecord]:
        """Get all records for a specific classification.

        Parameters
        ----------
        classification : str
            Classification label to filter by

        Returns
        -------
        records : list of ClassificationRecord
            Records matching the classification
        """
        return [
            record for record in self.records
            if record.classification == classification
        ]

    def get_statistics(self) -> dict:
        """Get comprehensive classification statistics.

        Returns
        -------
        stats : dict
            Dictionary containing:
            - 'total_classified': int - Total classifications
            - 'classifications': dict - Counts per classification
            - 'session_duration': str - Time since session started
            - 'start_time': str - Session start time
            - 'classifications_by_time': dict - Classifications over time

        """
        now = datetime.now()
        duration = now - self.session_start

        return {
            'total_classified': len(self.records),
            'classifications': self.get_all_classifications(),
            'session_duration': str(duration),
            'start_time': self.session_start.isoformat(),
            'session_start_time': self.session_start.isoformat(),
            'last_classification_time': (
                self.records[-1].timestamp.isoformat()
                if self.records else None
            ),
        }

    def clear_history(self) -> int:
        """Clear all classification history.

        Returns
        -------
        cleared_count : int
            Number of records that were cleared
        """
        count = len(self.records)
        self.records.clear()
        logger.info(f"Cleared {count} classification records")
        return count

    def save_session(self, output_file: Path) -> bool:
        """Save classification session to JSON file.

        Parameters
        ----------
        output_file : Path
            Path to file where session will be saved

        Returns
        -------
        success : bool
            True if session was saved successfully

        Raises
        ------
        IOError
            If unable to write to output file
        """
        output_file = Path(output_file)

        try:
            data = {
                'session_start': self.session_start.isoformat(),
                'records': [record.to_dict() for record in self.records],
                'statistics': self.get_statistics(),
            }

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(
                f"Session saved to {output_file} "
                f"({len(self.records)} records)"
            )
            return True

        except IOError as e:
            msg = f"Error saving session to {output_file}: {str(e)}"
            logger.error(msg)
            raise IOError(msg) from e

    @classmethod
    def load_session(cls, session_file: Path) -> 'Classifier':
        """Load classification session from JSON file.

        Parameters
        ----------
        session_file : Path
            Path to session file to load

        Returns
        -------
        classifier : Classifier
            Loaded classifier with session data

        Raises
        ------
        FileNotFoundError
            If session file does not exist
        ValueError
            If session file is invalid
        """
        session_file = Path(session_file)

        if not session_file.exists():
            msg = f"Session file not found: {session_file}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            with open(session_file, 'r') as f:
                data = json.load(f)

            classifier = cls()
            classifier.session_start = datetime.fromisoformat(
                data['session_start']
            )

            for record_data in data['records']:
                record = ClassificationRecord.from_dict(record_data)
                classifier.records.append(record)

            logger.info(
                f"Loaded session from {session_file} "
                f"({len(classifier.records)} records)"
            )
            return classifier

        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in session file: {str(e)}"
            logger.error(msg)
            raise ValueError(msg) from e
        except Exception as e:
            msg = f"Error loading session from {session_file}: {str(e)}"
            logger.error(msg)
            raise ValueError(msg) from e

    def export_summary(self, output_file: Path) -> bool:
        """Export classification summary as human-readable text.

        Parameters
        ----------
        output_file : Path
            Path to output summary file

        Returns
        -------
        success : bool
            True if export was successful

        Raises
        ------
        IOError
            If unable to write to output file
        """
        output_file = Path(output_file)

        try:
            stats = self.get_statistics()
            summary_lines = [
                "HiRISE Chip Analyzer - Classification Summary",
                "=" * 60,
                "",
                f"Session Start: {stats['start_time']}",
                f"Total Classified: {stats['total_classified']}",
                f"Session Duration: {stats['session_duration']}",
                "",
                "Classification Breakdown:",
                "-" * 60,
            ]

            for classification, count in sorted(
                stats['classifications'].items()
            ):
                percentage = (
                    100 * count / stats['total_classified']
                    if stats['total_classified'] > 0 else 0
                )
                summary_lines.append(
                    f"  {classification}: {count} ({percentage:.1f}%)"
                )

            summary_lines.extend([
                "",
                "=" * 60,
            ])

            with open(output_file, 'w') as f:
                f.write("\n".join(summary_lines))

            logger.info(f"Summary exported to {output_file}")
            return True

        except IOError as e:
            msg = f"Error exporting summary to {output_file}: {str(e)}"
            logger.error(msg)
            raise IOError(msg) from e
