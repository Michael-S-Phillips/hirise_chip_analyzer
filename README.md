# HiRISE Chip Analyzer

A GUI application for classifying HiRISE image chips for the presence of pit features.

## Installation

### Prerequisites
- Python 3.11+
- Conda (Anaconda or Miniconda)

### Setup

1. **Create and activate the conda environment:**
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n hirise_chip_analyzer python=3.11
conda activate hirise_chip_analyzer
```

2. **Install dependencies:**
```bash
pip install numpy pillow rasterio scikit-image scipy pyqt5
```

3. **Install the package (development mode):**
```bash
pip install -e .
```

## Launching the Application

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hirise_chip_analyzer
python -m hirise_chip_analyzer
```

## Quick Start

1. **Select Source Directory** - Choose folder containing TIFF images
2. **Select Output Directory** - Choose where to save classified images
3. **Load and Classify**:
   - View images one at a time
   - Click **Yes - Pits** or **No - No Pits**
   - Classified images are automatically removed from the list
4. **Export Results** - Save classification summary when done

## Image Enhancement Tools

Use these tools to improve pit visibility:

### Brightness & Contrast
- **Brightness**: Lighten (-100) or darken (+100) the image
- **Contrast**: Reduce (0) or increase (200) feature visibility

### Advanced Pit Detection Features
- **CLAHE**: Enhances local contrast (ideal for small pit structures)
- **Sharpen**: Makes pit edges more pronounced (0-100 strength)
- **Invert**: Converts dark pits to light features for alternative perspective

**Reset button** returns all adjustments to defaults.

## Best Practices for Pit Detection

### 1. Start with Default Settings
- Begin with original image to establish baseline
- Helps avoid over-processing artifacts

### 2. CLAHE for Small Pits
- Enable CLAHE first if pits are hard to see
- Enhances local contrast without losing detail
- Standard technique in remote sensing

### 3. Combine Adjustments
- Try CLAHE + slight sharpening for best pit visibility
- Use invert as alternative perspective for borderline cases
- Experiment to find optimal combination for your image set

### 4. Use Navigation Carefully
- **Previous/Next**: Browse without classifying
- **Skip**: Skip uncertain images for later review
- **Yes/No**: Only classify when confident

### 5. Save Sessions
- Use **File → Save Session** to save progress
- Use **File → Load Session** to resume later
- Useful for multi-session analysis

### 6. Review Before Export
- Check statistics before exporting
- Use **File → Export Summary** for detailed report

## Keyboard Workflow

| Action | Method |
|--------|--------|
| Navigate images | Previous/Next buttons or keyboard arrows |
| Classify image | Yes/No buttons |
| Skip uncertain | Skip button |
| Adjust appearance | Use slider controls on right panel |
| Reset adjustments | Click Reset button |

## Keyboard Shortcuts

- **Ctrl+O** - Open source directory
- **Ctrl+E** - Set output directory
- **Ctrl+S** - Save session
- **Ctrl+L** - Load session

## Output

Classified images are organized in directories:
```
output_directory/
├── pits/              # Images containing pits
└── no_pits/           # Images with no pits
```

## Features

✅ Load TIFF images (GIS-format and standard)
✅ Real-time image adjustments
✅ Advanced pit detection enhancements (CLAHE, sharpening, invert)
✅ Automatic file organization by classification
✅ Session save/load functionality
✅ Progress tracking and statistics
✅ Export classification summaries

## File Organization

```
src/hirise_chip_analyzer/
├── core/
│   ├── image_loader.py      # TIFF image I/O
│   ├── classifier.py        # Classification tracking
│   └── file_organizer.py    # File management
├── gui/
│   └── main_window.py       # PyQt5 GUI
└── __main__.py              # Application entry point
```

## Testing

Run tests to verify installation:
```bash
python -m pytest tests/ -v
```

All 96 tests should pass.

## Troubleshooting

### Images won't load
- Ensure TIFF files are in the selected directory
- Check file extensions (.tif or .tiff)
- For GIS-format TIFF, rasterio automatically handles conversion

### Adjustments not visible
- Try CLAHE or sharpening to enhance contrast
- Adjust brightness for very dark images
- Use invert to see alternative perspective

### Performance issues
- Close other applications
- Reduce image resolution if possible
- Adjustments apply in real-time - they're fast even on large images

## Development

- **Code style**: Black (88 char line length)
- **Type hints**: 100% coverage on function signatures
- **Documentation**: NumPy-style docstrings
- **Testing**: Comprehensive test suite (96 tests)

## License

© 2025 HiRISE Team

## Support

For issues or questions, check the CLAUDE.md file for development details or review the test suite for usage examples.
