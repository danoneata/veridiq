  ## Project Structure
  - `veridiq/` - Main package directory
  - `veridiq/data.py` - Data loading utilities (AV1M dataset)
  - `veridiq/utils.py` - General utility functions
  - `veridiq/fsfm/` - FSFM model implementations
  - `veridiq/show_predictions.py` - Main prediction and visualization module
  - `veridiq/scripts/` - Analysis and processing scripts

  ## Development Setup
  - Install in development mode: `pip install -e .`
  - Dependencies are managed via `requirements.txt`
  - Python 3.7+ required

  ## Common Commands
  - Install package: `pip install -e .`
  - Source enviornment: `conda activate veridiq`
  - Format code: `black <file_path>`

  ## Coding Standards
  - Use absolute imports with `veridiq.` prefix for internal modules
  - Use `black` to format the code.
  - Follow PEP 8 style guidelines
  - Use type hints where appropriate
  - Maintain defensive security practices - no malicious code

  ## Data Requirements
  - AV1M dataset located at `/data/av-deepfake-1m/av_deepfake_1m/`