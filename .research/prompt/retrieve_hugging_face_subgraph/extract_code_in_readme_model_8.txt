
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
language:
- ar
library_name: tf-keras
license: mit
metrics:
- accuracy
---

## Model description
This biLSTMhard model is a Bidirectional Long-Short-Term Memory (BiLSTM) architecture trained from scratch on the Hotel Arabic Reviews Dataset (HARD) dataset with one bidirectional LSTM layer with 150 hidden states. 
It achieves the following results on the evaluation test set: 
 - Accuracy: 0.8013



## BibTeX Citations:
```bash
@inproceedings{alshahrani-etal-2024-arabic,
    title = "{{A}rabic Synonym {BERT}-based Adversarial Examples for Text Classification}",
    author = "Alshahrani, Norah  and
      Alshahrani, Saied  and
      Wali, Esma  and
      Matthews, Jeanna",
    editor = "Falk, Neele  and
      Papi, Sara  and
      Zhang, Mike",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-srw.10",
    pages = "137--147",
    abstract = "Text classification systems have been proven vulnerable to adversarial text examples, modified versions of the original text examples that are often unnoticed by human eyes, yet can force text classification models to alter their classification. Often, research works quantifying the impact of adversarial text attacks have been applied only to models trained in English. In this paper, we introduce the first word-level study of adversarial attacks in Arabic. Specifically, we use a synonym (word-level) attack using a Masked Language Modeling (MLM) task with a BERT model in a black-box setting to assess the robustness of the state-of-the-art text classification models to adversarial attacks in Arabic. To evaluate the grammatical and semantic similarities of the newly produced adversarial examples using our synonym BERT-based attack, we invite four human evaluators to assess and compare the produced adversarial examples with their original examples. We also study the transferability of these newly produced Arabic adversarial examples to various models and investigate the effectiveness of defense mechanisms against these adversarial examples on the BERT models. We find that fine-tuned BERT models were more susceptible to our synonym attacks than the other Deep Neural Networks (DNN) models like WordCNN and WordLSTM we trained. We also find that fine-tuned BERT models were more susceptible to transferred attacks. We, lastly, find that fine-tuned BERT models successfully regain at least 2{\%} in accuracy after applying adversarial training as an initial defense mechanism.",
}
```
```bash
@misc{alshahrani2024arabic,
      title={{Arabic Synonym BERT-based Adversarial Examples for Text Classification}}, 
      author={Norah Alshahrani and Saied Alshahrani and Esma Wali and Jeanna Matthews},
      year={2024},
      eprint={2402.03477},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Training and evaluation data

We trained this model on a 90% training set and evaluated it on a 10% testing set.

## Training procedure
We have trained this model using the PaperSpace GPU-Cloud service. We used a machine with 8 CPUs, 45GB RAM, and A6000 GPU with 48GB RAM.
### Training hyperparameters

The following hyperparameters were used during training:

| Hyperparameters | Value |
| :-- | :-- |
train_batch_size:| 128
num_epochs: |15
dropout-1: |0.3
dropout-2: |0.3


 ## Model Plot

<details>
<summary>View Model Plot</summary>

![Model Image](./model.png)

</details>
Output:
{
    "extracted_code": ""
}
