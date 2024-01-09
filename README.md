# Decision Making For Multi-Robot Fixture Planning Using Multi-Agent Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="assets/mafp_architecture.png" width="700">
</p>

## Contents

- [Overview](#1)
- [Brief Synopsis](#2)
- [Manual Installation](#3)
- [Docker Container](#4)
- [Troubleshooting](#5)
- [Citing This Work](#6)

<a id='1'></a>

## Overview

This is the repository to go along with the paper "Decision Making For Multi-Robot Fixture Planning Using Multi-Agent Reinforcement Learning". This paper provides two representative models of an aerospace wing spar and wing panel and can be used for both training and inference.

This repo is split into two sections. The first requires the setup of the MATLAB Runtime and installation of Python packages in a virtual environment. This process is more complicated and may not work on older machines. The second (far easier) method is to use the Docker runtime and provided Docker image to run the training and inference process.

If you want to replicate our results, we provide network weights for each agent in the agent sets of 1 to 11 as in the work, with additional results regarding the determination of a Nash equilibrium in a single-stage game.

<a id='2'></a>

## Brief Synopsis

Fixture layout planning is the process of designing the layout for components undergoing a manufacturing task such as drilling or riveting ([Pehlivan & Summers, 2006](https://www.tandfonline.com/doi/abs/10.1080/00207540600865386)). In this process, the method aims to find positions for fixtures in such a way that they minimise any deformation or residual stresses that the component experiences during the task:
$$\underset{\tau}{\text{minimise }} |f_w(\tau)|$$
Traditional methods have relied on optimisation techniques that search for a global minima in fixture positions that minimise the experienced deformation. However, these optimisation methods frequently enter local minima and believe they have found the global solution.

Reinforcement learning is a machine learning technique that seeks to learn optimal behaviour by having an agent interact within an environment and learn which actions produce the best rewards ([Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html)). In the multi-agent setting, the agents are seeking to maximise the global value function of all agents:
$$V^n_{\pi^n, \boldsymbol{\pi}^{-n}}(s) = \mathbb{E}_{a_{t+1} \sim P, \boldsymbol{a}_t \sim \boldsymbol{\pi}} \left [ \sum_{t=0}^\infty \gamma^t R^n_t | s_0 = s\right ]$$
Due to the multi-agent setting, RL practitioners seek to embed game-theoretic guarantees in the learning stage of the agents. However, some instances only see global rewards returned to the agents, which leads to a field known as "team theory". Similar to game theory, it covers the cooperation of agents where the reward at each state is a function of the actions of all agents with no individual rewards. This leads to a player-by-player equilibrium, which is identical in nature to the Nash equilibrium ([van Schuppen, 2014](https://link.springer.com/chapter/10.1007/978-3-319-10407-2_18)):
$$J(\{a^*_n, \boldsymbol{a}^*_{-n}\}) \leq J(\{a_n, \boldsymbol{a}^*_{-n}\})$$

<a id='3'></a>

## Manual Installation

<a id='4'></a>

## Docker Container

<a id='5'></a>

## Troubleshooting

<a id='6'></a>

## Citing This Work

If you want to cite this work, please refer to our preprint on TechXriv:

```bibtex
 @article{marl_fixtures_preprint_2023, 
    type={preprint}, 
    title={Decision Making For Multi-Robot Fixture Planning Using Multi Agent Reinforcement Learning}, 
    DOI={10.36227/techrxiv.24171534.v1}, 
    publisher={TechRxiv}, 
    author={Canzini, Ethan and Auledas Noguera, Marc and Pope, Simon and Tiwari, Ashutosh}, 
    year={2023}, 
    month=oct, 
    language={en} 
 }
```

Any questions, please forward them to: <ecanzini1@sheffield.ac.uk>. If you want to use our approach and want advice, feel free to reach out! Pull requests for other functionality (new models, different RL algorithms, bug fixes etc.) are welcome.
