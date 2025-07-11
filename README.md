# Veridiq

A package for deepfake detection and verification using the AV1M dataset.

## Installation

Install in development mode:
```bash
pip install -e .
```

## Requirements

- Python 3.7+
- AV1M dataset located at `/data/av-deepfake-1m/av_deepfake_1m/`

## Project Structure

- `veridiq/` - Main package directory
- `veridiq/data.py` - Data loading utilities (AV1M dataset)
- `veridiq/utils.py` - General utility functions
- `veridiq/fsfm/` - FSFM model implementations
- `veridiq/show_predictions.py` - Main prediction and visualization module
- `veridiq/scripts/` - Analysis and processing scripts

## Usage

Activate the conda environment:
```bash
conda activate veridiq
```

## Development

Format code using:
```bash
black <file_path>
```
