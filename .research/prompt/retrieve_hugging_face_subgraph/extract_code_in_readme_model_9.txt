
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
language:
- pt
license: apache-2.0
library_name: keras
pipeline_tag: text-classification
tags:
- bilstm
- portuguese
- pt
- fake-news
- binary-classification
metrics:
- accuracy
- precision
- recall
- f1-score
datasets: vzani/corpus-fake-br
model-index:
- name: portuguese-fake-news-classifier-bilstm-fake-br
  results:
  - task:
      type: text-classification
    dataset:
      name: Fake.br
      type: vzani/corpus-fake-br
      split: test
    metrics:
    - name: accuracy
      type: accuracy
      value: 0.938889
    - name: precision_macro
      type: precision
      value: 0.938919
      args:
        average: macro
    - name: recall_macro
      type: recall
      value: 0.938889
      args:
        average: macro
    - name: f1_macro
      type: f1
      value: 0.938888
      args:
        average: macro
    - name: precision_weighted
      type: precision
      value: 0.938919
      args:
        average: weighted
    - name: recall_weighted
      type: recall
      value: 0.938889
      args:
        average: weighted
    - name: f1_weighted
      type: f1
      value: 0.938888
      args:
        average: weighted
    - name: n_test_samples
      type: num
      value: 1440
---
# BiLSTM for Fake News Detection (Portuguese)

## Model Overview

This repository contains a trained **BiLSTM** model for **fake news detection in Portuguese**.
The model was trained and evaluated on corpora derived from Brazilian Portuguese dataset **[Fake.br](https://github.com/roneysco/Fake.br-Corpus)**.

- **Architecture**: Bidirectional LSTM (Keras)
- **Task**: Binary text classification (Fake vs. True)
- **Language**: Portuguese (`pt`)
- **Framework**: Keras / TensorFlow
- **Training source code**: https://github.com/viniciuszani/portuguese-fake-new-classifiers

---

## Available Variants

- [**bilstm-combined**](https://huggingface.co/vzani/portuguese-fake-news-classifier-bilstm-combined)
  Fine-tuned using the [combined dataset](https://huggingface.co/datasets/vzani/corpus-combined) from Fake.br and FakeTrue.Br.

- [**bilstm-fake-br**](https://huggingface.co/vzani/portuguese-fake-news-classifier-bilstm-fake-br)
  Fine-tuned using the [Fake.br dataset](https://huggingface.co/datasets/vzani/corpus-fake-br) from Fake.br.

- [**bilstm-faketrue-br**](https://huggingface.co/vzani/portuguese-fake-news-classifier-bilstm-faketrue-br)
  Fine-tuned using the [FakeTrue.Br dataset](https://huggingface.co/datasets/vzani/corpus-faketrue-br) from FakeTrue.Br.

Each variant has its own confusion matrix, classification report, and predictions stored as artifacts.

---

## Training Details

```python
{
    "ngram_upper": 2,
    "units": 120,
    "dropout": 0.3374510345164157,
    "recurrent_dropout": 0.1588638491073387,
    "max_tokens": 96000,
    "embed_dim": 71,
    "embed_max_seq_len": 51,
    "learning_rate": 0.00011662663429277272,
    "batch_size": 16,
    "epochs": 8,
}
```

---

## Evaluation Results

Evaluation metrics are stored in the repo as:
- `confusion_matrix.png`
- `final_classification_report.parquet`
- `final_predictions.parquet`

These files provide per-class performance and prediction logs for reproducibility.

---

## How to Use

This model is a **Keras** model stored as `final_bilstm_model.keras`.

```python
import keras
import tensorflow as tf
from huggingface_hub import hf_hub_download

repo_id = "vzani/portuguese-fake-news-classifier-bilstm-fake-br"  # or combined / faketrue-br
filename = "final_bilstm_model.keras"

model_path = hf_hub_download(repo_id=repo_id, filename=filename)
model = keras.models.load_model(model_path)


def predict(text: str) -> tuple[bool, float]:
    input_data = tf.convert_to_tensor([[text]], dtype=tf.string)
    probs = model.predict(input_data)  # type: ignore
    prob = float(probs.flatten()[0])  # type: ignore
    pred = prob >= 0.5

    # Convert the probability in case of Fake
    prob = prob if pred else 1 - prob
    return pred, prob


if __name__ == "__main__":
    text = "BOMBA! A Dilma vai taxar ainda mais os pobres!"
    print(predict(text))
```

The expected output is a Tuple where the first entry represents the classification (`True` for true news and `False` for fake news) and the second the probability assigned to the predicted class (ranging from 0 to 1.0).
```
(False, 0.997499808203429)
```

## Source code

You can find the source code that produced this model in the repository below:
- https://github.com/viniciuszani/portuguese-fake-new-classifiers

The source contains all the steps from data collection, evaluation, hyperparameter fine tuning, final model tuning and publishing to HuggingFace.
If you use it, please remember to credit the author and/or cite the work.

## License

- Base model BERTimbau: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Fine-tuned models and corpora: Released under the same license for academic and research use.

## Citation

```bibtex
@misc{zani2025portuguesefakenews,
  author       = {ZANI, Vinícius Augusto Tagliatti},
  title        = {Avaliação comparativa de técnicas de processamento de linguagem natural para a detecção de notícias falsas em Português},
  year         = {2025},
  pages        = {61},
  address      = {São Carlos},
  school       = {Instituto de Ciências Matemáticas e de Computação, Universidade de São Paulo},
  type         = {Trabalho de Conclusão de Curso (MBA em Inteligência Artificial e Big Data)},
  note         = {Orientador: Prof. Dr. Ivandre Paraboni}
}
```

Output:
{
    "extracted_code": "import keras\nimport tensorflow as tf\nfrom huggingface_hub import hf_hub_download\n\nrepo_id = \"vzani/portuguese-fake-news-classifier-bilstm-fake-br\"  # or combined / faketrue-br\nfilename = \"final_bilstm_model.keras\"\n\nmodel_path = hf_hub_download(repo_id=repo_id, filename=filename)\nmodel = keras.models.load_model(model_path)\n\n\ndef predict(text: str) -> tuple[bool, float]:\n    input_data = tf.convert_to_tensor([[text]], dtype=tf.string)\n    probs = model.predict(input_data)  # type: ignore\n    prob = float(probs.flatten()[0])  # type: ignore\n    pred = prob >= 0.5\n\n    # Convert the probability in case of Fake\n    prob = prob if pred else 1 - prob\n    return pred, prob\n\n\nif __name__ == \"__main__\":\n    text = \"BOMBA! A Dilma vai taxar ainda mais os pobres!\"\n    print(predict(text))"
}
