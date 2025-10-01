
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: mit
tags:
- Imitation Learning
- Expert Trajectory
- InvertedPendulum-v2
pretty_name: InvertedPendulum-v2 Expert Dataset
size_categories:
- 1M<n<10M
---

# InvertedPendulum-v2 - Continuous Imitation Learning from Observation

This dataset was created for the paper Explorative imitation learning: A path signature approach for continuous environments.
It is based on InvertedPendulum-v2, which is an older version for the MuJoCo environment.
If you would like to use newer version, be sure to check: [IL-Datasets](https://github.com/NathanGavenski/IL-Datasets) repository for the updated list.

## Description

The dataset consists of 10 episodes with an average episodic reward of `1000.0`.
Each entry consists of:
```
obs (list): observation with length 2.
action (int): action (0 or 1).
reward (float): reward point for that timestep.
episode_starts (bool): if that state was the initial timestep for an episode.
```

## Usage

Feel free to download and use the `teacher.jsonl` dataset as you please.
If you are interested in using our PyTorch Dataset implementation, feel free to check the [IL Datasets](https://github.com/NathanGavenski/IL-Datasets/blob/main/src/imitation_datasets/dataset/dataset.py) project.
There, we implement a base Dataset that downloads this dataset and all other datasets directly from HuggingFace.
The Baseline Dataset also allows for more control over train and test splits and how many episodes you want to use (in cases where the 1k episodes are not necessary).

## Citation

```{bibtex}
@incollection{gavenski2024explorative,
	title={Explorative Imitation Learning: A Path Signature Approach for Continuous Environments},
	author={Gavenski, Nathan and Monteiro, Juarez and Meneguzzi, Felipe and Luck, Michael and Rodrigues, Odinaldo},
	booktitle={ECAI 2024},
	pages={}
	year={2024},
	publisher={IOS Press}
}
```
Output:
{
    "extracted_code": ""
}
