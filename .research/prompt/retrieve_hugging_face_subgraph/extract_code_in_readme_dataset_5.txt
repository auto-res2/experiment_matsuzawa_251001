
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: mit
task_categories:
- image-classification
language:
- en
tags:
- mnist
- image
- digit
- synthetic
- houdini
pretty_name: MNIST Bakery Dataset
size_categories:
- 10K<n<100K
---
# 🧁 MNIST Bakery Dataset

![digits samples](https://i.imgur.com/kcKvfR7.png)

A procedurally synthesized variant of the classic MNIST dataset, created using **SideFX Houdini** and designed for experimentation in **data augmentation**, **synthetic data generation**, and **model robustness research**. 
See the [ML-Research](https://github.com/atcarter714/ML-Research) repository on GitHub for Python notebooks, experiments and the Houdini scene files.

![houdini viewport](https://i.imgur.com/LZqdiKN.png)


---

## 🎯 Purpose

This dataset demonstrates how **procedural generation pipelines** in 3D tools like Houdini can be used to create **high-quality, synthetic training data** for machine learning tasks. It is intended for:

- Benchmarking model performance using synthetic vs. real data  
- Training models in **low-data** or **zero-shot** environments  
- Developing robust classifiers that generalize beyond typical datasets  
- Evaluating augmentation and generalization strategies in vision models

---

## 🛠️ Generation Pipeline

All data was generated using the `.hip` scene:  
```bash
./houdini/digitgen_v02.hip
```

## 🧪 Methodology

### 1. Procedural Digit Assembly
- Each digit `0`–`9` is generated using a random font in each frame via Houdini’s **Font SOP**.
- Digits are arranged in a clean **8×8 grid**, forming sprite sheets with **64 digits per render**.

### 2. Scene Variability
- Fonts are randomly selected per frame.
- Procedural distortions are applied including:
  - Rotation
  - Translation
  - Skew
  - Mountain noise displacement
- This ensures high variability across samples.

### 3. Rendering
- Scene renders are executed via **Mantra** or **Karma**.
- Output format: **grayscale 224×224 px** sprite sheets (`.exr` or `.jpg`).

![stage/](https://i.imgur.com/LqPks5t.png)


### 4. Compositing & Cropping
- A **COP2 network** slices the sprite sheet into **28×28** digit tiles.
- Each tile is labeled by its original digit and saved to:
```
./output/0/img_00001.jpg
./output/1/img_00001.jpg
...
```

![compositing](https://i.imgur.com/mAnMQuB.png)


### 🧾 Dataset Structure

```bash
mnist_bakery_data/
├── 0/
│   ├── img_00001.jpg
│   ├── ...
├── 1/
│   ├── img_00001.jpg
│   └── ...
...
├── 9/
│   └── img_00001.jpg
```


- All images: grayscale `.jpg`, 28×28 resolution
- Total: **40,960 samples**
- ~4,096 samples per digit

---

## 📊 Statistics

| Set       | Samples | Mean    | StdDev   |
|-----------|---------|---------|----------|
| MNIST     | 60,000  | 0.1307  | 0.3081   |
| Synthetic | 40,960  | 0.01599 | 0.07722  |

> Combine mean/std using weighted averaging if mixing both datasets.

---

## 📚 Usage Example

```python
from torchvision import transforms, datasets

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.01599], std=[0.07722])  # Approximate weighted normalization
])

dataset = datasets.ImageFolder('./mnist_bakery_data', transform=transform)
```

---

#### 🧠 Credits

**Author**: Aaron T. Carter

**Organization**: Arkaen Solutions

**Tools Used**: Houdini, PyTorch, PIL

___


Output:
{
    "extracted_code": "from torchvision import transforms, datasets\n\ntransform = transforms.Compose([\n    transforms.Grayscale(),\n    transforms.ToTensor(),\n    transforms.Normalize(mean=[0.01599], std=[0.07722])  # Approximate weighted normalization\n])\n\ndataset = datasets.ImageFolder('./mnist_bakery_data', transform=transform)"
}
