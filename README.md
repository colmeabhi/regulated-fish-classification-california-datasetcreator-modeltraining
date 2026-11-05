# California Fish Classification

An automated pipeline for building and training fish classification models to help identify California recreational fishing species and their regulatory status.

## Overview

This project builds a comprehensive dataset and trains deep learning models to classify fish species into regulatory categories (prohibited, restricted, legal) for California recreational fishing compliance.

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Build Dataset
Run the automated pipeline to collect and process data:
```bash
./build_dataset.sh
```

This will:
- Download images from iNaturalist and FishBase
- Generate synthetic images for underrepresented species
- Preprocess and validate all images
- Create train/validation/test splits, **grouped by iNaturalist observation** so that several photos of the same fish never land in different splits

### 3. Train Models

**Basic CNN:**
```bash
python scripts/07_train.py
```

**Transfer Learning (ResNet50):**
```bash
python scripts/08_train_transfer_learning.py
```

**Hierarchical Classification:**
```bash
python scripts/10_train_hierarchical_classification.py
```

### 4. Make Predictions

```bash
python scripts/11_predict_hierarchical.py path/to/fish_image.jpg
```

### 5. Evaluate on the Test Set

```bash
python scripts/12_evaluate_hierarchical.py            # held-out test split
python scripts/12_evaluate_hierarchical.py --split val
```

This runs the full two-stage pipeline (category → species) on every image and writes to `evaluation_results/`:

- `hierarchical_<split>_report.md`: headline metrics, per-category and per-species recall, all with 95% confidence intervals
- `hierarchical_<split>_confusion.png`: category and species confusion matrices
- `hierarchical_<split>_critical_errors.csv`: every prohibited fish that was **not** predicted as prohibited
- `hierarchical_<split>_metrics.json`: all of the above in machine-readable form

The key safety metric is **prohibited → legal**: the share of zero-take fish that the app would tell an angler they can keep. The report also includes a threshold sweep ("flag as prohibited if P(prohibited) ≥ t") that trades false alarms on legal fish for fewer missed prohibited fish. Choose `t` on the validation split, then report it on test.

## Dataset

**Target:** ~5,500 images across 27 species

**Categories:**
- 12 Prohibited species (zero take)
- 11 Restricted species (sub-bag limits)
- 4 Legal species (comparison)

**Data Sources:**
- iNaturalist (citizen science observations)
- FishBase (scientific database)
- Synthetic images (Google Gemini 2.5 Flash)
- Image Library (local collections)

## Project Structure

```
├── scripts/
│   ├── 01_download_inaturalist.py      # Download from iNaturalist
│   ├── 02_download_fishbase.py         # Download from FishBase
│   ├── 03_import_image_library.py      # Import local images
│   ├── 04_generate_synthetic_images.py # Generate synthetic data
│   ├── 05_preprocess.py                # Preprocess images
│   ├── 06_create_splits.py             # Create train/val/test splits
│   ├── 07_train.py                     # Train basic CNN
│   ├── 08_train_transfer_learning.py   # Train ResNet50
│   ├── 10_train_hierarchical_classification.py  # Train hierarchical model
│   ├── 11_predict_hierarchical.py      # Make predictions
│   └── 12_evaluate_hierarchical.py     # Test-set evaluation + safety metrics
├── data/
│   ├── raw/                            # Downloaded images
│   ├── processed/                      # Preprocessed images
│   └── metadata/                       # Dataset metadata
├── models/                             # Trained models
└── config.yaml                         # Species configuration

```

## Models

**1. Basic CNN** - Simple 5-block convolutional network
**2. ResNet50** - Transfer learning from ImageNet
**3. Hierarchical** - Two-stage classification (category → species)

| Model | Val accuracy | Test accuracy |
|---|---|---|
| Basic CNN (no aug) | 57.4% | 56.4% |
| Basic CNN (aug) | 54.4% | 56.0% |
| ResNet50 (no aug) | 85.9% | 85.5% |
| ResNet50 (aug) | 89.9% | not yet measured |
| Hierarchical (2-stage) | 91.2% | not yet measured; run `12_evaluate_hierarchical.py` |

> **Note:** these figures were produced with the earlier per-image split. iNaturalist observations often contain several photos of the same fish, so near-duplicates could appear in both train and val/test, which inflates accuracy. Rebuild the splits (`05_preprocess.py`, `06_create_splits.py`) and retrain before reporting final test numbers.

## License

This dataset is compiled for research and educational purposes. Individual images retain their original licenses (CC-BY, CC0, CC-BY-NC).

## Acknowledgments

Data sources include iNaturalist community observations, FishBase scientific database, and synthetic images generated using Google Gemini AI.
