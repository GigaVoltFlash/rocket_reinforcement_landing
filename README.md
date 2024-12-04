# Testing RL policies for Rocket Landing in Different Simulated Wind Conditions
The goal of this project was to develop an RL policy trained to land a rocket at a desired location on the surface. The rocket can control itself using its gimballed rocket engine. 

The simulation environment for this project was taken from Zhengxia Zou's rocket-recycling project {https://github.com/jiupinjia/rocket-recycling}, which does a a great job at modeling a simplified version of the rocket dynamics involved, setting up a discrete action table, and a human viewable rendering of the system.

The reward function was inspired by some of Zou's work, although noticeable modifications were made.

A DQN policy with continuous state space and a discrete action space was used. This DQN policy uses an MLP to represent the Q function with two layers in its network.
