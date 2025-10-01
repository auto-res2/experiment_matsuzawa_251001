
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
annotations_creators: []
language: en
license: mit
size_categories:
- 10K<n<100K
task_categories:
- image-classification
task_ids: []
pretty_name: curated-mnist
tags:
- fiftyone
- image
- image-classification
description: 'Curated MNIST dataset with train/val/test splits, including predictions of different models: CLIP, LeNet, and retrained LeNet.'
split_field: tags
dataset_name: curated-mnist
organization: andandandand
commit_message: Add curated MNIST dataset with train/val/test splits
dataset_summary: |
  ![image/png](dataset_preview.webp)

  This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 70000 samples.

  ## Installation

  If you haven't already, install FiftyOne:

  ```bash
  pip install -U fiftyone
  ```

  ## Usage

  ```python
  import fiftyone as fo
  from fiftyone.utils.huggingface import load_from_hub

  # Load the dataset
  # Note: other available arguments include 'max_samples', etc
  dataset = load_from_hub("andandandand/curated-mnist")

  # Launch the App
  session = fo.launch_app(dataset)
  ```
---

# Dataset Card for CuratedMNIST

This dataset is a curated version of the classic MNIST dataset, enriched with model predictions, embeddings, and various analytical fields generated using the [FiftyOne](https://github.com/voxel51/fiftyone) library. It was created as part of the "Image Classification and Dataset Curation with FiftyOne and PyTorch" tutorial to demonstrate practical computer vision workflows.

The dataset contains the original 60,000 training and 10,000 test samples, with the training set further split into training (85%) and validation (15%) subsets. It includes predictions from OpenAI's CLIP model (zero-shot), a custom-trained LeNet-5 model, and a final LeNet-5 model that was fine-tuned on an augmented set of challenging samples. This structure makes it ideal for educational purposes, allowing users to explore model comparison, error analysis, and data-centric AI techniques.

![image/png](dataset_preview.webp)

This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 70000 samples.

## Installation

If you haven't already, install FiftyOne:

```bash
pip install -U fiftyone
```

## Usage

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the dataset
# Note: other available arguments include 'max_samples', etc
dataset = load_from_hub("Voxel51/curated-mnist")

# Launch the App
session = fo.launch_app(dataset)
```

### Dataset Description

This curated version of MNIST serves as a comprehensive resource for learning and experimenting with image classification workflows. Starting with the original MNIST dataset, this version adds multiple layers of metadata and analysis to each sample. It is designed to accompany a Voxel51 tutorial focused on dataset curation and model evaluation.

The key additions include:
*   **Train/Validation/Test Splits:** The original 60,000 training images are split into a training set (51,000 samples) and a validation set (9,000 samples). The original 10,000 test images are maintained as the test set.
*   **Model Predictions:** Each sample in the appropriate split contains predictions from different models:
    *   Zero-shot predictions from OpenAI's CLIP (`clip-vit-base32-torch`).
    *   Predictions from a baseline LeNet-5 CNN trained from scratch on the training set.
    *   Predictions from a retrained LeNet-5 model, fine-tuned on a curated set of misclassified and challenging samples with data augmentation.
*   **Embeddings:** Both image embeddings from CLIP and feature embeddings from the trained LeNet-5 model are included.
*   **Analysis Fields:** FiftyOne's "Brain" methods were used to compute `hardness`, `mistakenness`, `uniqueness`, and `representativeness` scores, enabling deep dives into data quality and model behavior.

- **Curated by:** Antonio Rueda-Toicen
- **Funded by:** Voxel51
- **Language(s) (NLP):** en
- **License:** mit

### Dataset Sources

- **Repository:** [andandandand/practical-computer-vision](https://github.com/andandandand/practical-computer-vision)
- **Paper:**
    - For MNIST: [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
    - For CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **Source:**
    - [Notebook on GitHub](https://github.com/andandandand/practical-computer-vision/blob/main/notebooks/Image_Classification_and_Dataset_Curation_with_FiftyOne_and_PyTorch_Getting_Started.ipynb) 
    - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andandandand/practical-computer-vision/blob/main/notebooks/Image_Classification_and_Dataset_Curation_with_FiftyOne_and_PyTorch_Getting_Started.ipynb)


### Direct Use

This dataset is primarily intended for educational and research purposes in computer vision and data-centric AI. Suitable uses include:
*   Learning how to load, visualize, and interact with image classification datasets using FiftyOne.
*   Comparing the performance of different classification models (e.g., zero-shot vs. supervised, baseline vs. fine-tuned).
*   Understanding and applying techniques for error analysis, such as identifying hard samples, potential label errors (`mistakenness`), and outliers (`uniqueness`).
*   Exploring the concept of image embeddings for visualization (PCA, UMAP) and similarity search.
*   Practicing data curation workflows, such as identifying problematic samples for augmentation and retraining.
*   Benchmarking new classification algorithms or data augmentation strategies on a well-understood but richly annotated dataset.

### Out-of-Scope Use

This dataset is not intended for training production-level models for general-purpose digit recognition, as it is based on the relatively simple and constrained MNIST domain (28x28 grayscale images). Models trained on this dataset will not generalize to real-world images of digits with complex backgrounds, colors, or fonts. The dataset should not be used to draw conclusions about model performance on natural images.

## Dataset Structure

The dataset contains three splits, identified by the `tags` field on each sample: `train`, `validation`, and `test`.

Each sample contains the following fields:
*   `filepath`: The path to the 28x28 grayscale PNG image.
*   `ground_truth`: A `fiftyone.Classification` object containing the ground truth digit label (0-9).
*   `tags`: A list of tags indicating the split (`train`, `validation`, or `test`).
*   `metadata`: A `fiftyone.Metadata` object with image properties like width, height, and size in bytes.
*   `clip_embeddings`: A 512-dimensional vector from the `clip-vit-base32-torch` model. (Test set only)
*   `clip_zero_shot_classification`: A `fiftyone.Classification` object with the zero-shot prediction from CLIP. (Test set only)
*   `lenet_embeddings`: An 84-dimensional feature vector from the penultimate layer of the trained LeNet-5 model. (Train set only)
*   `lenet_classification`: A `fiftyone.Classification` object with the prediction from the baseline LeNet-5 model. (Test set only)
*   `lenet_train_classification`: Predictions from the LeNet-5 model on the training data, used for analysis. (Train set only)
*   `retrained_lenet_classification`: Predictions from the final fine-tuned LeNet-5 model. (Test set only)
*   `hardness`: A float score indicating how challenging the sample was for a model to classify.
*   `mistakenness`: A float score indicating the likelihood of a sample being mislabeled, based on model disagreement with the ground truth.
*   `uniqueness`: A float score measuring how distinct a sample's embedding is from others.
*   `representativeness`: A float score measuring how well a sample's embedding represents its local cluster.


### Curation Rationale

The primary motivation for creating this dataset was to provide a hands-on, practical resource for a tutorial on modern computer vision workflows. While MNIST is a "solved" problem, its familiarity makes it an excellent canvas for demonstrating advanced techniques. By enriching MNIST with model predictions, embeddings, and analytical scores, the dataset enables users to move beyond simple model training and focus on the iterative process of data curation, model evaluation, and error analysis that is central to building robust machine learning systems.


#### Data Collection and Processing

The source data is the Modified National Institute of Standards and Technology (MNIST) dataset. The curation process was performed programmatically as follows:
1.  The official `train` and `test` splits of MNIST were loaded from the FiftyOne Dataset Zoo.
2.  The 60,000-sample `train` split was deterministically divided into an 85% training set (51,000 samples) and a 15% validation set (9,000 samples).
3.  A modern variant of the LeNet-5 architecture was trained from scratch in PyTorch on the new training set, with the best model checkpoint selected based on validation loss.
4.  The trained LeNet-5 model and a pre-trained `clip-vit-base32-torch` model from the FiftyOne Model Zoo were applied to the test set to generate predictions (`lenet_classification`, `clip_zero_shot_classification`).
5.  The LeNet-5 model was also run on the training set to identify misclassified samples and samples classified with low confidence.
6.  Analysis fields (embeddings, hardness, mistakenness, etc.) were computed using FiftyOne Brain methods.
7.  A subset of problematic training samples (misclassified, unique, highly representative, or low-confidence) were identified for data augmentation.
8.  The baseline LeNet-5 model was fine-tuned on a new dataset composed of the original training data plus augmented versions of the problematic samples.
9.  The final, retrained model was applied to the test set to generate `retrained_lenet_classification` predictions.

#### Who are the source data producers?

The original MNIST data was collected by the National Institute of Standards and Technology. The dataset was later processed and popularized by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. The handwritten digits were sourced from two groups: American Census Bureau employees and American high school students.

### Annotations

The dataset contains two types of annotations:
1.  **Ground Truth Labels**: These are the original digit labels (0-9) from the MNIST dataset. The annotators were the original writers of the digits (Census Bureau employees and high school students).
2.  **Model Predictions**: These are generated labels from the CLIP, LeNet-5, and retrained LeNet-5 models. The "annotators" in this case are the machine learning models themselves. These predictions are stored in separate fields and are intended for model evaluation and comparison, not as ground truth.

The curation process used `mistakenness` scores to identify samples where the ground truth label is likely incorrect, but these labels were not corrected in order to preserve the original dataset benchmark.

## Bias, Risks, and Limitations

*   **Label Noise:** As discussed in the curation notebook and academic literature, the original MNIST dataset contains a small number of ambiguous or incorrectly labeled samples. This dataset's `mistakenness` field helps identify these, but they have not been removed or corrected.
*   **Lack of Diversity:** MNIST consists of small, centered, grayscale handwritten digits. It is not representative of digits in natural scenes ("in the wild"). Models trained on this dataset will perform poorly on other visual domains without significant adaptation.
*   **Demographic Bias:** The handwriting styles are sourced from American Census Bureau employees and high school students from the 1990s. The dataset may not be representative of global handwriting styles or handwriting from different demographic groups.
*   **Model-Specific Biases:** The added predictions and embeddings reflect the specific architectures and training data of the models used (CLIP, LeNet-5). The `lenet_` fields are specific to the model trained in the tutorial and may not generalize to other architectures.

## Citation

**BibTeX:**

To cite the original MNIST dataset:
```bibtex
@article{lecun1998gradient,
  title={Gradient-based learning applied to document recognition},
  author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  journal={Proceedings of the IEEE},
  volume={86},
  number={11},
  pages={2278--2324},
  year={1998},
  publisher={IEEE}
}
```

To cite the CLIP model:
```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
```

## Dataset Card Authors

Antonio Rueda-Toicen

## Dataset Card Contact

[GitHub: andandandand](https://github.com/andandandand)
Output:
{
    "extracted_code": "import fiftyone as fo\nfrom fiftyone.utils.huggingface import load_from_hub\n\n# Load the dataset\n# Note: other available arguments include 'max_samples', etc\ndataset = load_from_hub(\"andandandand/curated-mnist\")\n\n# Launch the App\nsession = fo.launch_app(dataset)\n\nimport fiftyone as fo\nfrom fiftyone.utils.huggingface import load_from_hub\n\n# Load the dataset\n# Note: other available arguments include 'max_samples', etc\ndataset = load_from_hub(\"Voxel51/curated-mnist\")\n\n# Launch the App\nsession = fo.launch_app(dataset)"
}
