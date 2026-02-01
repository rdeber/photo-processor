# Stock Photo Processor

Automated stock photography processing pipeline for Adobe Stock, Shutterstock, and iStock.

## Features

- **RAW Processing**: Supports Canon CR2, Google Pixel DNG, and standard JPEG/PNG
- **Auto Straighten**: Automatic horizon, vertical, and perspective correction
- **Auto Exposure**: Automatic white balance and brightness/contrast correction
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

## Setup

### API Key

Create a `.env` file in the project root with your Claude API key:

```bash
ANTHROPIC_API_KEY=sk-ant-xxxxx
```

Or copy the example file:

```bash
cp .env.example .env
# Edit .env with your API key
```

## CLI Reference

### Basic Usage

```bash
stock-process <input> [OPTIONS]
```

`<input>` can be a single image file or a directory of images.

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output` | `-o` | `./processed` | Output directory for processed images |
| `--brightness` | `-b` | `0.45` | Target brightness (0-1 scale) |
| `--contrast` | | `1.1` | Contrast strength (1.0 = no change) |
| `--straighten` | `-s` | `auto` | Geometry: auto (H+V+perspective), horizontal, vertical, none |
| `--min-size` | | `4000` | Minimum dimension in pixels |
| `--config` | `-c` | | Path to custom config file |
| `--no-face-blur` | | | Disable face detection and blurring |
| `--no-logo-removal` | | | Disable logo/trademark removal |

### Examples

```bash
# Process a single image
stock-process image.CR2

# Process a folder with custom output directory
stock-process ./raw-photos/ --output ./processed/

# Brighten dark images
stock-process image.jpg --brightness 0.55

# Add more contrast punch
stock-process image.jpg --contrast 1.2

# Combine brightness and contrast adjustments
stock-process ./photos/ -o ./out -b 0.5 --contrast 1.15

# Only level horizon (no vertical correction)
stock-process image.jpg --straighten horizontal

# Only fix leaning verticals (buildings, trees)
stock-process image.jpg --straighten vertical

# Disable auto-straighten
stock-process image.jpg -s none

# Skip face blurring (e.g., for landscape photos)
stock-process ./landscapes/ --no-face-blur

# Skip logo removal (faster processing)
stock-process ./photos/ --no-logo-removal

# Use a custom config file
stock-process ./photos/ --config my-settings.yaml
```

## Configuration

Settings can be configured via YAML file. The default config is at `config/default.yaml`.

### All Options

| Option | Default | Description |
|--------|---------|-------------|
| **Output** | | |
| `output_quality` | `100` | JPEG quality (1-100, 100 = no compression) |
| `min_dimension` | `4000` | Minimum pixels on longest edge |
| **Face Blurring** | | |
| `face_blur` | `true` | Enable face detection and blurring |
| `face_blur_strength` | `25` | Gaussian blur kernel size |
| **Logo Removal** | | |
| `remove_logos` | `true` | Detect and remove logos/trademarks |
| `remove_signs` | `true` | Detect and remove store signs/text |
| **Image Processing** | | |
| `sharpen_amount` | `1.0` | Unsharp mask intensity (0 = none) |
| `denoise_strength` | `5` | Non-local means denoising (0 = none) |
| **Auto Exposure** | | |
| `auto_white_balance` | `true` | Gray world white balance correction |
| `auto_exposure` | `true` | Automatic brightness/contrast adjustment |
| `target_brightness` | `0.45` | Target mean brightness (0-1 scale) |
| `contrast_strength` | `1.1` | Contrast multiplier (1.0 = no change) |
| **Geometry** | | |
| `straighten_mode` | `auto` | auto (H+V+perspective), horizontal, vertical, none |
| **Metadata** | | |
| `keywords_min` | `42` | Minimum keywords to generate |
| `keywords_max` | `47` | Maximum keywords to generate |
| `author_name` | `"Ryan DeBerardinis"` | Photographer name for IPTC Creator |
| `copyright_holder` | `"Ryan DeBerardinis"` | Name for copyright notice |

### Example Config File

```yaml
# my-settings.yaml
output_quality: 100
min_dimension: 4000

# Brighter output for dark indoor shots
auto_white_balance: true
auto_exposure: true
target_brightness: 0.50
contrast_strength: 1.15

# Geometry - auto levels horizon + verticals + perspective
straighten_mode: auto

# Processing
sharpen_amount: 1.0
denoise_strength: 5

# Your info
author_name: "Your Name"
copyright_holder: "Your Name"
```

Use with: `stock-process ./photos/ --config my-settings.yaml`

## Supported Formats

### Input
- Canon CR2 (RAW)
- Google Pixel DNG (RAW)
- JPEG
- PNG
- TIFF

### Output
- JPEG (quality 100, sRGB color space)
- Sidecar JSON with full metadata

## Requirements

- Python 3.11+
- Claude API key (for metadata generation)

## License

MIT
