# A Reinforcement Learning-based Volt-VAR Control Dataset
This repository contains a suite of Volt-VAR control (VVC) benchmarks for conducting research on sample efficient, safe, and robust RL-based VVC algorithms. It includes the IEEE 13, 123, and 8500-node test feeders wrapped as a gym-like environment along with baseline algorithm implementations to reproduce the results of [[1]](#1)

## Installation

1. Download .zip of this repository
2. Install the packages
```
pip install -r environment.txt
```

## Usage
### Run baseline algorithms
Set environment and algorithm parameters in the ```config``` variable in ```main.py```, then run the program
```
python main.py
```
### Implement custom off-policy RL algorithms
Create implement the ```update()```, ```act_deterministic```, and ```act_probabilistic``` functions in the ```/algos/template.py``` file

### Use the environment
```
from policy_train import train_policy
from policy_eval import eval_policy
config = {
    "setup": {
        "env": "env_13",
        "reward_format": 'volt_dev'},
    "algo": {
        "algo": "dqn",
        "dims_hidden_neurons": (120, 120),
        "scale_reward": 5.0,
        "discount": 0.95,
        "batch_size": 128,
        "lr": 0.0005,
        "copy_steps": 30,
        "eps_len": 500,
        "eps_max": 1.0,
        "eps_min": 0.02,
        "training_steps": 1000},
    "replay": {
        "size": 15660,
    },
    "seed": 0,
    "device": "cpu",
}
train_res= train_policy(config)
eval_res = eval_policy(config, train_res)
```
## References
<a id="1">[1]</a> 
Gao, Y., Yu, N., (2021). A Reinforcement Learning-based Volt-VAR Control Dataset and Environment.

## Citing this benchmark
To cite this benchmark, please cite the following paper:

```
@article{gao2021benchmark,
  title={A Reinforcement Learning-based Volt-VAR Control Dataset and Environment},
  author={Gao, Yuanqi and Yu, Nanpeng},
}
```
