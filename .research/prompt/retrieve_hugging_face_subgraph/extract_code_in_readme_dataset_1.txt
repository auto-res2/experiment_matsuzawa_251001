
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: apache-2.0
task_categories:
- image-classification
- unconditional-image-generation
language:
- en
size_categories:
- 10K<n<100K
---
# MNIST 32×32 RGB Dataset

A preprocessed version of the classic MNIST dataset with handwritten digits, upscaled to 32×32 pixels and converted to RGB format for enhanced deep learning applications.

## Overview

This dataset takes the original MNIST handwritten digit dataset and transforms it for modern computer vision tasks:
- **Original format**: 28×28 grayscale images
- **Enhanced format**: 32×32 RGB images
- **Digit labels**: 0-9 (single-label classification)
- **Image format**: RGB PIL Images

## Dataset Statistics

### Training Set (60,000 samples)
| Digit | Count | Digit | Count |
|-------|-------|-------|-------|
| 0     | 5,923 | 5     | 5,421 |
| 1     | 6,742 | 6     | 5,918 |
| 2     | 5,958 | 7     | 6,265 |
| 3     | 6,131 | 8     | 5,851 |
| 4     | 5,842 | 9     | 5,949 |

### Test Set (10,000 samples)
| Digit | Count | Digit | Count |
|-------|-------|-------|-------|
| 0     | 980   | 5     | 892   |
| 1     | 1,135 | 6     | 958   |
| 2     | 1,032 | 7     | 1,028 |
| 3     | 1,010 | 8     | 974   |
| 4     | 982   | 9     | 1,009 |

## Directory Structure

```
mnist_32_c3/
├── README.md                               # This documentation
├── train-00000-of-00001.parquet            # Training data (Parquet format)
└── test-00000-of-00001.parquet             # Test data (Parquet format)
```

## Key Features

- **Upscaled Resolution**: Enhanced from 28×28 to 32×32 pixels for better feature extraction
- **RGB Format**: Converted from grayscale to RGB (3-channel) format
- **PIL Integration**: Images loaded as PIL RGB objects ready for preprocessing
- **Standard Splits**: Maintains original MNIST train/test division
- **HuggingFace Compatible**: Full integration with datasets library
- **Efficient Loading**: Parquet format for fast columnar data access and compression

## Usage

### Loading with HuggingFace Datasets

```python
from datasets import load_dataset

# Load the dataset using the custom script
dataset = load_dataset("FrankCCCCC/mnist_32_c3", trust_remote_code=True)

print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# Access a sample
sample = dataset['train'][0]
print(f"Image shape: {sample['image'].size}")    # (32, 32)
print(f"Image mode: {sample['image'].mode}")     # RGB
print(f"Label: {sample['label']}")               # Integer: 0-9
```

## Transformations Applied

The dataset preprocessing pipeline includes:

1. **Format Conversion**: IDX → Parquet
2. **Channel Expansion**: Grayscale → RGB (L → RGB)
3. **Resolution Upscaling**: 28×28 → 32×32 (using PIL resize)
4. **Data Storage**: Parquet format for efficient storage and loading
5. **Data Type**: PIL Image objects for easy integration

## Dataset Format

Each sample contains:
```python
{
    'image': PIL.Image,    # 32×32 RGB image
    'label': int,          # Digit class (0-9)
}
```

## Technical Details

- **Original Source**: MNIST Database of Handwritten Digits
- **Format**: Parquet files for efficient columnar storage
- **Preprocessing**: Grayscale → RGB conversion, 28×28 → 32×32 upscaling
- **Loading**: HuggingFace Datasets with custom GeneratorBasedBuilder
- **Compression**: Parquet format provides built-in compression and fast access

## Citation

```bibtex
@article{lecun1998mnist,
  title={The MNIST database of handwritten digits},
  author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  year={1998},
  url={http://yann.lecun.com/exdb/mnist/}
}

@misc{mnist32c3,
  title={MNIST 32×32 RGB Dataset},
  author={Enhanced MNIST for Modern Deep Learning},
  year={2024},
  note={Preprocessed version of original MNIST with RGB format and 32×32 resolution}
}
```

## License

This dataset follows the same license as the original MNIST dataset. The original MNIST database is available under the Creative Commons Attribution-Share Alike 3.0 license.

## Acknowledgments

- Based on the original MNIST dataset by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- Enhanced for modern deep learning applications with RGB format and increased resolution
- Compatible with HuggingFace Datasets ecosystem for seamless integration
- Optimized for CNN architectures and transfer learning applications
Output:
{
    "extracted_code": "from datasets import load_dataset\n\n# Load the dataset using the custom script\ndataset = load_dataset(\"FrankCCCCC/mnist_32_c3\", trust_remote_code=True)\n\nprint(f\"Train samples: {len(dataset['train'])}\")\nprint(f\"Test samples: {len(dataset['test'])}\")\n\n# Access a sample\nsample = dataset['train'][0]\nprint(f\"Image shape: {sample['image'].size}\")    # (32, 32)\nprint(f\"Image mode: {sample['image'].mode}\")     # RGB\nprint(f\"Label: {sample['label']}\")               # Integer: 0-9"
}
