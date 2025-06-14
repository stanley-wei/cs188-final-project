# CS 188 Final Project
![nut_assembly.png](https://github.com/stanley-wei/cs188-final-project/blob/main/nut_assembly.png?raw=true)

**Group**: Stanley Wei & Ryu Adams

For our final project for UCLA's COM SCI 188: Introduction to Robotics course, we implemented two methods for imitation learning from expert demonstrations:

1. DMP: Cartesian-space dynamic motion primitives.
2. NN: Behavior cloning using a neural network.

Evaluated on the robosuite Nut Assembly task:

| Method | Training Time | Success Rate | Avg. Time
| -------- | ------- | ------- | ------- | 
| DMP  | 378 ms. (1 demo.) | 0.76 | 6.389 sec.
| NN | 3.2 min (200 demos) | 0.72 | 3.481 sec.

You can see our project website here: https://sites.google.com/g.ucla.edu/stanley-ryu-cs188-project

## Setup

This project requires `robosuite` to be installed. Additionally, `torch` is required to run the neural network components.

To test the DMP policy, run:
```
cd dmp && python3 test_dmp.py
```
- Note: The `DMPPolicy` class requires a file containing demonstrations to learn from!

To test the NN policy, run:
```
cd nn && python3 test_nn.py
```
- To train your own NN policy, use `behavior_cloning.py`!
