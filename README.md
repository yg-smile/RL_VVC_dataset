# A Reinforcement Learning-based Volt-VAR Control Dataset
This repository contains a suite of Volt-VAR control (VVC) benchmarks for conducting research on sample efficient, safe, and robust RL-based VVC algorithms. It includes the IEEE 13, 123, and 8500-node test feeders wrapped as a gym-like environment along with baseline algorithm implementations to reproduce the results of [[1]](#1)

## Installation

1. Download .zip of this repository
2. Install the packages

TODO

## Usage
### Run baseline algorithms
Set environment and algorithm parameters in the ```config``` variable in ```main.py```, then run the program
```
python main.py
```
### Implement custom off-policy RL algorithms
Create implement the ```update()```, ```act_deterministic```, and ```act_probabilistic``` functions in the ```/algos/template.py``` file

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
