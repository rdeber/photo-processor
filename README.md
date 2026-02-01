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
| `--straighten-sensitivity` | | `1.0` | Line detection sensitivity (0.5=strict, 2.0=loose) |
| `--straighten-max-angle` | | `15` | Maximum rotation angle in degrees |
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

# Fine-tune straighten for complex scenes (more aggressive detection)
stock-process image.jpg --straighten-sensitivity 1.5

# Conservative straightening (only very confident corrections)
stock-process image.jpg --straighten-sensitivity 0.5 --straighten-max-angle 5
```

## Auto-Straighten Algorithm

The auto-straighten feature automatically corrects tilted horizons, leaning verticals, and perspective distortion using computer vision. Here's how it works:

### How It Works

The algorithm uses a two-pass approach:

**Pass 1: Rotation Correction**
1. **Edge Detection**: Uses Canny edge detection to find edges in the image
2. **Line Detection**: Uses Hough transform to find straight lines
3. **Line Classification**: Lines are classified as horizontal (within 15° of level) or vertical (within 15° of upright)
4. **Length-Weighted Averaging**: Each line's angle is weighted by its pixel length. Longer lines (building edges, door frames) have more influence than short lines (window frames, fire escape details)
5. **Rotation**: The image is rotated to correct the weighted average tilt
6. **Cropping**: Border artifacts are cropped away

**Pass 2: Perspective Correction** (auto mode only)
1. **Vertical Line Analysis**: Vertical lines are separated into left-half and right-half of the image
2. **Convergence Detection**: Measures if verticals are converging (typical of looking up at buildings)
3. **Perspective Transform**: Applies a transform to make verticals parallel

### Straighten Modes

| Mode | Description |
|------|-------------|
| `auto` | Full correction: horizontal leveling + vertical straightening + perspective correction |
| `horizontal` | Only level the horizon (useful for landscapes, seascapes) |
| `vertical` | Only straighten vertical lines (useful for architecture with intentional horizon tilt) |
| `none` | Skip all geometry correction |

### Tuning Parameters

#### `straighten_sensitivity` (default: 1.0)

Controls how aggressively lines are detected:

| Value | Behavior |
|-------|----------|
| `0.5` | **Strict**: Requires longer, more prominent lines. Better for noisy scenes with lots of small details |
| `1.0` | **Normal**: Balanced detection for typical photos |
| `1.5-2.0` | **Loose**: Detects more/shorter lines. Better for images with few structural elements |

How sensitivity affects detection:
- **Canny edge thresholds**: Lower sensitivity = higher thresholds = fewer edges detected
- **Minimum line length**: Lower sensitivity = longer minimum = only prominent lines
- **Hough threshold**: Lower sensitivity = higher threshold = stricter line detection

#### `straighten_max_angle` (default: 15.0)

Maximum rotation angle the algorithm will apply. If the calculated correction exceeds this, it's skipped entirely (assumes the image is intentionally tilted or the algorithm misdetected).

| Value | Use Case |
|-------|----------|
| `5.0` | Conservative: Only correct small tilts, preserve artistic angles |
| `15.0` | Normal: Correct most accidental tilts |
| `25.0` | Aggressive: Correct even significant tilts |

### Troubleshooting

**Image not being straightened:**
- The algorithm may not detect enough lines. Try `--straighten-sensitivity 1.5` to detect more lines
- The correction may be too small (< 0.05°) and is being skipped
- The correction may exceed `max_angle` and is being skipped

**Over-correction / Wrong direction:**
- The algorithm may be detecting non-structural lines (fire escapes, window details). Try `--straighten-sensitivity 0.5` to focus on longer lines only
- For complex urban scenes, the median of detected lines may not represent true vertical. Consider using `--straighten horizontal` for just horizon leveling

**Too much cropping:**
- Larger rotations require more cropping to remove border artifacts
- Use `--straighten-max-angle 5` to limit rotation and preserve more of the image

### Example Workflows

```bash
# Architecture photography (prioritize vertical correction)
stock-process buildings.jpg --straighten auto --straighten-sensitivity 0.7

# Landscape photography (just level the horizon)
stock-process landscape.jpg --straighten horizontal

# Complex urban scene (conservative, long lines only)
stock-process street.jpg --straighten-sensitivity 0.5 --straighten-max-angle 10

# Trust the algorithm fully (aggressive correction)
stock-process photo.jpg --straighten-sensitivity 1.5 --straighten-max-angle 20
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
| **Geometry (Auto-Straighten)** | | |
| `straighten_mode` | `auto` | auto (H+V+perspective), horizontal, vertical, none |
| `straighten_sensitivity` | `1.0` | Line detection: 0.5=strict (long lines), 2.0=loose (more lines) |
| `straighten_max_angle` | `15.0` | Maximum rotation angle in degrees (skip if exceeded) |
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

# Geometry correction
straighten_mode: auto
straighten_sensitivity: 1.0    # 0.5=strict, 2.0=loose
straighten_max_angle: 15.0     # Max degrees of rotation

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
