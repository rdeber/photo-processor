# Stock Photo Processor - Project Plan

## Overview

A local CLI-based image processing pipeline for stock photography workflow automation. Takes raw camera files or JPEGs, processes them for quality while maintaining a photorealistic look, removes identifiable faces and logos/trademarks, generates AI-powered metadata, and outputs upload-ready JPEGs.

**Target platforms:** Adobe Stock, Shutterstock, iStock

## Requirements

### Input
- Canon CR2 raw files (primary camera)
- Google Pixel DNG raw files
- JPEG source files (Pixel smartphone)
- Batch processing support (~100 images/week typical volume)
- Original files remain untouched in place

### Output
- JPEG format, quality 100 no compression
- sRGB color space
- Minimum 4000px on longest edge
- AI-generated metadata:
  - **Title:** Concise, descriptive, searchable
  - **Description:** Thorough, unique description of each photo
  - **Keywords:** 42-47 stock-site optimized tags per image

### Processing Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  RAW/JPEG   │───▶│ Traditional │───▶│ AI Content  │───▶│ AI Metadata │───▶│   Export    │
│   Input     │    │ Processing  │    │   Removal   │    │ Generation  │    │   JPEGs     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                   │
                   ┌──────┴──────┐     ┌──────┴──────┐
                   │ NON-AI:     │     │ AI-POWERED: │
                   │ • Auto WB   │     │ • Face blur │
                   │ • Auto exp  │     │ • Logo/TM   │
                   │ • Sharpen   │     │   removal   │
                   │ • Denoise   │     │ • Sign text │
                   │ • Resize    │     │   removal   │
                   │ • Color/sRGB│     │             │
                   └─────────────┘     └─────────────┘
```

### Processing Philosophy

**Photorealism is paramount.** The final images must look natural, not AI-processed.

- **Traditional tools only** for core image processing (sharpen, denoise, resize, color correction). No AI upscaling or AI enhancement that creates the "fake" look.
- **AI tools permitted** for content removal (faces, logos, trademarks, store signs) but applied carefully to maintain natural appearance.
- **Content to remove:**
  - Identifiable faces → blur/anonymize
  - Brand logos (on cars, clothing, products)
  - Store names and signage
  - Trademarks and copyrighted imagery

## Tech Stack

### Core Language
- **Python 3.11+** - Best ecosystem for image processing and AI integration

### RAW Processing
- `rawpy` - RAW file decoding (LibRaw wrapper, supports CR2 and DNG)
- Camera-specific color profiles for accurate rendering

### Traditional Image Processing (Non-AI)
- `Pillow` / `pillow-simd` - Core image manipulation, resize, crop
- `opencv-python` - Sharpening (unsharp mask), denoising (non-local means)
- `colour-science` - Accurate sRGB color space conversion
- Standard algorithms only - no AI enhancement/upscaling

### Face Detection & Blurring (AI-Assisted)
- `face_recognition` library (dlib-based, high accuracy)
- Gaussian blur applied to detected face regions
- Configurable blur strength to maintain natural look

### Logo/Trademark Removal (AI-Assisted, Two-Pass)
1. **Identification:** Claude Vision API analyzes image, returns list of problematic content
   - Brand logos, store names, signage, trademarks
   - Returns descriptions like "Nike swoosh on t-shirt", "Starbucks sign in window"
2. **Localization:** GroundingDINO takes those descriptions and finds bounding boxes
3. **Segmentation:** SAM (Segment Anything Model) creates pixel-perfect masks from boxes
4. **Inpainting:** LaMa inpainting removes content seamlessly, photorealistic results

### AI Metadata Generation
- **Claude Vision API** (claude-sonnet-4-20250514)
  - Analyze processed image
  - Generate title, description, 42-47 keywords in one call
  - Optimized prompts for stock photography terminology
  - Estimated cost: ~$0.02-0.04 per image

### User Interface
- **CLI only** with `rich` for progress bars and formatted output
- Simple: `stock-process <input-dir> --output <output-dir>`

### Output & Metadata
- JPEG with quality 100 (no compression artifacts)
- Embedded IPTC metadata:
  - Title, description, keywords
  - **Author/Creator:** Photographer's name (configurable)
  - **Copyright:** © {YEAR} - {Photographer Name}
- Sidecar JSON files with full metadata for reference
- CSV summary of batch for easy review
- No camera EXIF data preserved (not needed)

## Project Structure

```
stock-photo-processor/
├── src/
│   └── stock_processor/
│       ├── __init__.py
│       ├── cli.py              # Command-line interface
│       ├── pipeline.py         # Main processing orchestration
│       ├── raw_handler.py      # RAW file decoding (CR2, DNG)
│       ├── image_processor.py  # Traditional processing (sharpen, denoise, resize)
│       ├── face_blur.py        # Face detection and blurring
│       ├── logo_remover.py     # Logo/trademark detection and removal
│       ├── metadata.py         # AI metadata generation (Claude Vision)
│       └── exporter.py         # JPEG export with embedded IPTC metadata
├── config/
│   └── default.yaml        # Default processing settings
├── tests/
├── pyproject.toml
└── README.md
```

## CLI Interface (Proposed)

```bash
# Process a single image
stock-process image.CR2

# Process entire folder
stock-process ./raw-photos/ --output ./processed/

# Process with custom settings
stock-process ./raw-photos/ --no-face-blur --min-size 5000

# Skip logo removal (if image has no street scenes, etc.)
stock-process ./raw-photos/ --no-logo-removal
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `output_quality` | 100 | JPEG quality (maximum, no compression) |
| `min_dimension` | 4000 | Minimum pixels on longest edge |
| `face_blur` | true | Auto-detect and blur faces |
| `face_blur_strength` | 25 | Gaussian blur kernel size |
| `remove_logos` | true | Detect and remove logos/trademarks |
| `remove_signs` | true | Detect and remove store signs/text |
| `sharpen_amount` | 1.0 | Unsharp mask intensity (subtle) |
| `denoise_strength` | 5 | Non-local means denoising (conservative) |
| `auto_white_balance` | true | Gray world white balance correction |
| `auto_exposure` | true | Automatic brightness/contrast adjustment |
| `target_brightness` | 0.45 | Target mean brightness (0-1 scale) |
| `contrast_strength` | 1.1 | Contrast multiplier (1.0 = no change) |
| `straighten_mode` | auto | Geometry correction: auto, horizontal, vertical, none |
| `keywords_min` | 42 | Minimum keywords to generate |
| `keywords_max` | 47 | Maximum keywords to generate |
| `author_name` | Ryan DeBerardinis | Photographer name for IPTC Creator field |
| `copyright_holder` | Ryan DeBerardinis | Name for copyright notice |

## Decisions Made

- [x] **Stock sites:** Adobe Stock, Shutterstock, iStock (all three)
- [x] **Keywords:** 42-47 per image, high-converting, descriptive
- [x] **Watermarking:** Not needed
- [x] **Logo/trademark handling:** Two-pass detection + removal (Claude Vision → GroundingDINO → SAM → LaMa)
- [x] **Preview mode:** Not needed - batch process and review output
- [x] **Cameras:** Canon DSLR (CR2), Google Pixel (DNG + JPEG)
- [x] **UI:** CLI only for now
- [x] **EXIF preservation:** Not needed - only embed new IPTC metadata
- [x] **Author/Copyright:** Include photographer name + copyright in metadata

## Open Questions

- [x] **Photographer name:** Ryan DeBerardinis
- [ ] **Keyword examples:** User will provide examples of their keywording strategy later

## Development Phases

### Phase 1: Core Pipeline (MVP) ✅ COMPLETE
- [x] Project setup (pyproject.toml, dependencies)
- [x] RAW file loading (CR2, DNG) and JPEG input support
- [x] Traditional image processing (sharpen, denoise, resize)
- [x] Auto white balance (gray world algorithm)
- [x] Auto exposure (brightness/contrast adjustment)
- [x] Auto straighten (horizon leveling, vertical correction)
- [x] sRGB color space conversion
- [x] JPEG export with quality settings
- [x] Basic CLI interface

### Phase 2: Content Removal 🔄 IN PROGRESS
- [x] Face detection and blurring (OpenCV fallback + optional face_recognition)
- [ ] Logo/trademark detection via Claude Vision
- [ ] Inpainting/removal of detected content
- [ ] Quality checks to ensure photorealistic output

### Phase 3: AI Metadata ✅ COMPLETE
- [x] Claude Vision API integration for analysis
- [x] Title generation (concise, descriptive)
- [x] Description generation (thorough, unique)
- [x] Keyword generation (42-47 stock-optimized tags)
- [x] IPTC metadata embedding

### Phase 4: Polish
- [x] Batch progress tracking with rich CLI output
- [ ] Configuration file support (YAML loading)
- [ ] Error handling and recovery (resume failed batches)
- [ ] CSV summary export

## Cost Estimates

| Component | Cost |
|-----------|------|
| Claude Vision - Logo detection (per image) | ~$0.02 |
| Claude Vision - Metadata generation (per image) | ~$0.02 |
| **Total per image** | ~$0.04 |
| Weekly batch (100 images) | ~$4.00 |
| Monthly estimate | ~$16.00 |

All image processing runs locally - no cloud compute costs. Only API costs are for Claude Vision calls.

## Development Workflow

### Git Strategy
- **Frequent commits:** Small, logical commits at each milestone
- **Push regularly:** Keep GitHub repo up to date throughout development
- **NO CO-AUTHOR LINES:** Never add "Co-Authored-By" to commit messages. Ryan is the sole author.
- **Commit checkpoints:**
  - Project scaffolding (pyproject.toml, structure)
  - Each module as it becomes functional
  - Integration milestones
  - Bug fixes and refinements

### Commit Milestone Plan
1. `Initial project setup` - pyproject.toml, folder structure, dependencies
2. `Add RAW file handling` - CR2/DNG loading with rawpy
3. `Add traditional image processing` - sharpen, denoise, resize, sRGB
4. `Add JPEG export` - quality 100, basic CLI working
5. `Add face detection and blurring` - face_recognition integration
6. `Add logo detection` - Claude Vision API integration
7. `Add logo localization` - GroundingDINO integration
8. `Add logo segmentation and removal` - SAM + LaMa inpainting
9. `Add AI metadata generation` - titles, descriptions, keywords
10. `Add IPTC embedding` - author, copyright, full metadata
11. `Add batch processing and progress` - rich CLI output
12. `Add configuration file support` - YAML config loading
13. `Polish and error handling` - recovery, edge cases
