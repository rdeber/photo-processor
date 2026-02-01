# Stock Photo Processor

Automated stock photography processing pipeline for Adobe Stock, Shutterstock, and iStock.

## Features

- **RAW Processing**: Supports Canon CR2, Google Pixel DNG, and standard JPEG/PNG
- **Traditional Image Processing**: Sharpen, denoise, resize (non-AI, photorealistic)
- **Face Anonymization**: Automatic face detection and blurring
- **Logo Removal**: AI-powered detection and removal of logos, trademarks, and signs
- **AI Metadata**: Generates titles, descriptions, and 42-47 optimized keywords
- **IPTC Embedding**: Embeds metadata directly into output JPEGs

## Installation

```bash
# Clone the repository
git clone https://github.com/rdeber/photo-processor.git
cd photo-processor

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install base package
pip install -e .

# Optional: Install face detection support
pip install -e ".[face]"

# Optional: Install logo removal support (large download)
pip install -e ".[logo-removal]"

# Optional: Install everything
pip install -e ".[full]"
```

## Usage

```bash
# Process a single image
stock-process image.CR2

# Process a folder of images
stock-process ./raw-photos/ --output ./processed/

# Process without face blurring
stock-process ./raw-photos/ --no-face-blur

# Process without logo removal
stock-process ./raw-photos/ --no-logo-removal
```

## Configuration

Copy `config/default.yaml` and customize settings:

```yaml
output_quality: 100
min_dimension: 4000
face_blur: true
face_blur_strength: 25
remove_logos: true
keywords_min: 42
keywords_max: 47
author_name: "Your Name"
copyright_holder: "Your Name"
```

## Requirements

- Python 3.11+
- Claude API key (for metadata generation)

## License

MIT
