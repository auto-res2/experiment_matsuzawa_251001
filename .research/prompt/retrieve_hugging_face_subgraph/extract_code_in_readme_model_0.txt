
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
tags:
- CartPole-v1
- deep-reinforcement-learning
- reinforcement-learning
- custom-implementation
library_name: cleanrl
model-index:
- name: DQN
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: CartPole-v1
      type: CartPole-v1
    metrics:
    - type: mean_reward
      value: 500.00 +/- 0.00
      name: mean_reward
      verified: false
---

# (CleanRL) **DQN** Agent Playing **CartPole-v1**

This is a trained model of a DQN agent playing CartPole-v1.
The model was trained by using [CleanRL](https://github.com/vwxyzjn/cleanrl) and the most up-to-date training code can be
found [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/CP_DDQN.py).

## Get Started

To use this model, please install the `cleanrl` package with the following command:

```
pip install "cleanrl[CP_DDQN]"
python -m cleanrl_utils.enjoy --exp-name CP_DDQN --env-id CartPole-v1
```

Please refer to the [documentation](https://docs.cleanrl.dev/get-started/zoo/) for more detail.


## Command to reproduce the training

```bash
curl -OL https://huggingface.co/pfunk/CartPole-v1-CP_DDQN-seed111/raw/main/dqn.py
curl -OL https://huggingface.co/pfunk/CartPole-v1-CP_DDQN-seed111/raw/main/pyproject.toml
curl -OL https://huggingface.co/pfunk/CartPole-v1-CP_DDQN-seed111/raw/main/poetry.lock
poetry install --all-extras
python dqn.py --track --wandb-entity pfunk --wandb-project-name dqpn --capture-video true --save-model true --upload-model true --hf-entity pfunk --exp-name CP_DDQN --double-learning --seed 111
```

# Hyperparameters
```python
{'alg_type': 'dqn.py',
 'batch_size': 256,
 'buffer_size': 300000,
 'capture_video': True,
 'cuda': True,
 'double_learning': True,
 'end_e': 0.1,
 'env_id': 'CartPole-v1',
 'exp_name': 'CP_DDQN',
 'exploration_fraction': 0.2,
 'gamma': 1.0,
 'hf_entity': 'pfunk',
 'learning_rate': 0.0001,
 'learning_starts': 1000,
 'max_gradient_norm': inf,
 'save_model': True,
 'seed': 111,
 'start_e': 1.0,
 'target_network_frequency': 100,
 'target_tau': 1.0,
 'torch_deterministic': True,
 'total_timesteps': 500000,
 'track': True,
 'train_frequency': 1,
 'upload_model': True,
 'wandb_entity': 'pfunk',
 'wandb_project_name': 'dqpn'}
```
    
Output:
{
    "extracted_code": "{'alg_type': 'dqn.py',\n 'batch_size': 256,\n 'buffer_size': 300000,\n 'capture_video': True,\n 'cuda': True,\n 'double_learning': True,\n 'end_e': 0.1,\n 'env_id': 'CartPole-v1',\n 'exp_name': 'CP_DDQN',\n 'exploration_fraction': 0.2,\n 'gamma': 1.0,\n 'hf_entity': 'pfunk',\n 'learning_rate': 0.0001,\n 'learning_starts': 1000,\n 'max_gradient_norm': inf,\n 'save_model': True,\n 'seed': 111,\n 'start_e': 1.0,\n 'target_network_frequency': 100,\n 'target_tau': 1.0,\n 'torch_deterministic': True,\n 'total_timesteps': 500000,\n 'track': True,\n 'train_frequency': 1,\n 'upload_model': True,\n 'wandb_entity': 'pfunk',\n 'wandb_project_name': 'dqpn'}"
}
