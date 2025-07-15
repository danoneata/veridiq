# Veridiq

A package for deepfake detection using the AV1M dataset.

## Installation

Install in development mode:
```bash
pip install -e .
```

Download linear models on CLIP and FSFM features:
```bash
wget https://github.com/danoneata/veridiq/releases/download/v0.0.1/output.zip
unzip output.zip
```

## Requirements

- Python 3.7+
- AV1M dataset located at `/data/av-deepfake-1m/av_deepfake_1m/`
- AV1M CLIP features located at `/data/av1m-test/other/CLIP_features/test`
- AV1M FSFM features located at `/data/audio-video-deepfake/FSFM_face_features/test_face`

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

Visualize explanations:
```bash
streamlit run --server.port 8080 veridiq/show_predictions.py
```

## Development

Format code using:
```bash
black <file_path>
```
