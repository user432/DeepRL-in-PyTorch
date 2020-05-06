# About Deep Reinforcement Learning

Reinforcement Learning is a machine learning approach for teaching agents how to solve tasks by trial and error. The combination of Reinforcement Learning and Deep Learning produces a series of important algorithms. This project will focus on referring to
relevant papers and implementing relevant algorithms as far as possible.

This repo aims to implement Deep Reinforcement Learning algorithms using [Pytorch](https://pytorch.org/).


## 1.Why do this?

- Implementing all of this algorithms really helps you with your **parameter tuning**.
- The coding process allows you to **better understand** the **principles** of the algorithm.

## 2.Lists of Algorithms

| No. | Status | Algorithm | Paper |
| --- | ------- | --------- | ----- |
| 1 | :green_check_mark:  | [DQN](/1.DQN) | [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) |
| 2 | :white_check_mark:  | [Double DQN](/2.Double%20DQN) | [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) |
| 3 | :white_check_mark:  | [Dueling DQN](/3.Dueling%20DQN) | [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) |
| 4 | :white_check_mark: | [REINFORCE](/4.REINFORCE) | [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) |
| 5 | <ul><li>- [ ] </li></ul>  | A3C + GAE | [High Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) |
| 6 | :white_check_mark: | [A2C](/6.A2C) | [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) |
| 7 | <ul><li>- [ ] </li></ul> | DPG | [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf) |
| 8 | <ul><li>- [ ] </li></ul>  | DDPG | [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) |
| 9 | <ul><li>- [ ] </li></ul>  | TRPO | [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) |
| 10 | <ul><li>- [ ] </li></ul>  | PPO | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) |
| 11 | <ul><li>- [ ] </li></ul> | ACTKR |  |
| 12 | <ul><li>- [ ] </li></ul>  | SAC | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf) |
| 13 | <ul><li>- [ ] </li></ul>  | SAC Alpha | [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf) |
| 14 | <ul><li>- [ ] </li></ul>  | TD3(Twin Delayed DDPG) | [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) |


### 3.Project Dependencies

- Python >=3.6
- PyTorch >= 1.3.1
- OpenAI Gym


### 4.Run

Each algorithm is implemented in a single package including:
```
main.py --A minimal executable example for algorithm  
[algorithm].py --Main body for algorithm implementation   
```
You can run algorithm from the  `main.py` w.r.t each algorithm's folder
- You can simply type `python main.py --help` in the algorithm package to view all parameters.

### 5.Best RL courses

- [Berkeley Deep RL](http://rll.berkeley.edu/deeprlcourse/)
- [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
- [David Silver's course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
