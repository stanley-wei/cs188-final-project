# CS 188 Final Project

For our final project for UCLA's COM SCI 188: Introduction to Robotics course, we implemented two methods for imitation learning from expert demonstrations:

1. DMP: Cartesian-space dynamic motion primitives.
2. NN: Behavior cloning using a neural network.

Evaluated on the robosuite Nut Assembly task:

| Method | Training Time | Success Rate | Avg. Time
| -------- | ------- | ------- | ------- | 
| DMP  | 378 ms. (1 demo.) | 0.76 | 0.021 sec.
| NN | sdfsdf | 0.76 | sdfsdf

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
