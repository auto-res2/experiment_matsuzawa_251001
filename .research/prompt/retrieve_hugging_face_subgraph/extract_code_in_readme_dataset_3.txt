
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: apache-2.0
task_categories:
- image-classification
language:
- en
size_categories:
- 10K<n<100K
---
# Colored MNIST Dataset

A comprehensive dataset of MNIST digits with RGB colored backgrounds, designed for multi-objective classification tasks including digit recognition and background color classification.

![Blue Example](images/blue_312.png) ![Green Example](images/green_8218.png) ![Red Example](images/red_8428.png)

## Overview

This dataset extends the classic MNIST digit dataset by adding colored backgrounds (blue, green, red) to create a multi-label classification challenge. Each image contains:
- **Digit label**: 0-9 (digit recognition)  
- **Color label**: blue, green, red (background color classification)
- **Image format**: 28×28 RGB PNG files

## Dataset Statistics

### Training Set (8,835 samples)
| Digit | Count | Digit | Count |
|-------|-------|-------|-------|
| 0     | 868   | 5     | 764   |
| 1     | 1,000 | 6     | 893   |
| 2     | 858   | 7     | 948   |
| 3     | 909   | 8     | 835   |
| 4     | 883   | 9     | 877   |

### Test Set (10,000 samples)
| Digit | Count | Digit | Count |
|-------|-------|-------|-------|
| 0     | 980   | 5     | 892   |
| 1     | 1,135 | 6     | 958   |
| 2     | 1,032 | 7     | 1,028 |
| 3     | 1,010 | 8     | 974   |
| 4     | 982   | 9     | 1,009 |

### Color Distribution
- **Blue**: ~2,946 training samples
- **Green**: ~2,968 training samples  
- **Red**: ~2,921 training samples

## Directory Structure

```
colored_mnist/
├── README.md
├── colored_mnist.py          # HuggingFace dataset loader
├── images/                   # Example images
│   ├── blue_312.png         # Blue background, digit 3
│   ├── green_8218.png       # Green background, digit 8  
│   └── red_8428.png         # Red background, digit 4
├── training/                 # Training data
│   ├── 0/                   # Digit 0 samples
│   │   ├── blue_1015.png
│   │   ├── green_1000.png
│   │   └── red_1028.png
│   └── ...                  # Digits 1-9
├── testing/                  # Test data
│   ├── 0/                   # Digit 0 samples
│   │   ├── blue_10.png
│   │   ├── green_1084.png
│   │   └── red_1001.png
│   └── ...                  # Digits 1-9
├── training.h5              # HDF5 format (optional)
└── testing.h5               # HDF5 format (optional)
```

## Usage

### Loading with HuggingFace Datasets

```python
from datasets import load_dataset

# Load the dataset using the custom script
dataset = load_dataset("FrankCCCCC/colored_mnist", trust_remote_code=True)

print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# Access a sample
sample = dataset['train'][0]
print(f"Image shape: {sample['image'].shape}")  # PIL.Image: Raw Image with shape (28, 28, 3)
print(f"Digit Label: {sample['digit_label']}")  # Integer: Digit class 0-9
print(f"Color Label: {sample['color_label']}")  # String: Color class: 'blue', 'green', 'red'
print(f"filename: {sample['filename']}")        # String: file name
print(f"image_path: {sample['image_path']}")    # String: file path
```

## Applications

This dataset is ideal for:

1. **Multi-task Learning**: Train models to simultaneously classify digits and colors
2. **Domain Adaptation**: Study how background colors affect digit recognition
3. **Bias Analysis**: Investigate color-digit correlations and model robustness
4. **Transfer Learning**: Pre-train on one task (digit/color) and fine-tune on another
5. **Curriculum Learning**: Start with single-task then progress to multi-task
6. **Attention Mechanisms**: Visualize what models focus on (digit vs. background)

## Features

- **Multi-label Classification**: Both digit and color labels for each image
- **Balanced Colors**: Approximately equal distribution across blue, green, red
- **Standard Format**: Compatible with PyTorch, TensorFlow, and HuggingFace
- **PIL Integration**: Images loaded as PIL objects for easy preprocessing
- **Metadata**: Includes original filename and image path information

## Dataset Format

Each sample contains:
```python
{
    'image': PIL.Image,           # 28×28 RGB image
    'digit_label': int,           # Digit class (0-9)
    'color_label': str,           # Color class ('blue'|'green'|'red')
    'filename': str,              # Original filename
    'image_path': str             # Full path to image file
}
```

## Requirements

```bash
pip install datasets pillow torch torchvision matplotlib numpy
```

## Citation

```bibtex
@misc{colorized-mnist,
  title={Colorized MNIST},
  author={Original MNIST Dataset},
  year={2024},
  url={https://github.com/jayaneetha/colorized-MNIST}
}
```

## License

This dataset follows the same license as the original MNIST dataset. Please cite both the original MNIST work and this colored extension when using this dataset.

## Acknowledgments

- Based on the original MNIST dataset by Yann LeCun et al.
- Color augmentation creates additional research opportunities for multi-task learning
- Compatible with modern deep learning frameworks and HuggingFace ecosystem
Output:
{
    "extracted_code": "from datasets import load_dataset\n\n# Load the dataset using the custom script\ndataset = load_dataset(\"FrankCCCCC/colored_mnist\", trust_remote_code=True)\n\nprint(f\"Train samples: {len(dataset['train'])}\")\nprint(f\"Test samples: {len(dataset['test'])}\")\n\n# Access a sample\nsample = dataset['train'][0]\nprint(f\"Image shape: {sample['image'].shape}\")  # PIL.Image: Raw Image with shape (28, 28, 3)\nprint(f\"Digit Label: {sample['digit_label']}\")  # Integer: Digit class 0-9\nprint(f\"Color Label: {sample['color_label']}\")  # String: Color class: 'blue', 'green', 'red'\nprint(f\"filename: {sample['filename']}\")        # String: file name\nprint(f\"image_path: {sample['image_path']}\")    # String: file path"
}
