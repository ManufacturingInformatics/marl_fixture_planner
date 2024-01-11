# Decision Making For Multi-Robot Fixture Planning Using Multi-Agent Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="assets/mafp_architecture.png" width="700">
</p>

## Contents

- [Overview](#1)
- [Brief Synopsis](#2)
- [Manual Installation](#3)
  - [MATLAB Runtime Installation](#3a)
  - [Package Installation](#3b)
  - [Training](#3c)
  - [Inference](#3d)
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

Fixture layout planning is the process of designing the layout for components undergoing a manufacturing task such as drilling or riveting ([Pehlivan & Summers, 2006](https://www.tandfonline.com/doi/abs/10.1080/00207540600865386)). In this process, the method aims to find positions for fixtures $A^*$ in such a way that they minimise any deformation or residual stresses that the component experiences during the task:

```math
A^* \in \argmin_{A \subseteq \mathcal{A}} |f_w(\tau)|
```

Traditional methods have relied on optimisation techniques that search for a global minima in fixture positions that minimise the experienced deformation. However, these optimisation methods frequently enter local minima and believe they have found the global solution.

Reinforcement learning is a machine learning technique that seeks to learn optimal behaviour by having an agent interact within an environment and learn which actions produce the best rewards ([Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html)). In the multi-agent setting infinite-horizon setting, the agents are seeking to maximise their individual value function with respect to the other agents:

```math
V^n_{\pi^n, \boldsymbol{\pi}^{-n}}(s) = \mathbb{E}_{a_{t+1} \sim P, \boldsymbol{a}_t \sim \boldsymbol{\pi}} \left [ \sum_{t=0}^\infty \gamma^t R^n_t | s_0 = s\right ]
```

Due to the multi-agent setting, RL practitioners seek to embed game-theoretic guarantees in the learning stage of the agents. However, some instances only see global rewards returned to the agents, which leads to a field known as "team theory". Similar to game theory, it covers the cooperation of agents where the reward at each state is a function of the actions of all agents with no individual rewards. This leads to a player-by-player equilibrium, which is identical in nature to the Nash equilibrium ([van Schuppen, 2014](https://link.springer.com/chapter/10.1007/978-3-319-10407-2_18)):

```math
J(\{a^*_n, \boldsymbol{a}^*_{-n}\}) \leq J(\{a_n, \boldsymbol{a}^*_{-n}\})
```

When using robotic fixture elements, the elements can be reconfigured to multiple different drilling tasks and can find optimal fixture plans that can reduce deformation across multiple different positions.

<p align="center">
  <img src="assets/Multi-Robot Fixtures For Milling.png" width="400">
</p>

In this work we use reinforcement learning alongside team theory to create a mulit-agent framework for determining optimal fixture placement for multiple drilling tasks on aerospace components.

<a id='3'></a>

## Manual Installation

This work is built on two pillars: an FEA simulator developed in MATLAB and a multi-agent reinforcement learning process developed in Python. The system specs that the model was trained on is as follows:

```
OS: Ubuntu 20.04 LTS
Python Version: 3.10.10
GPU: NVIDIA GeForce RTX 3080
CPU: Intel Core i9-10920X 12C/24T
RAM: 64 GB
```

This work has only been tested on Ubuntu - there is currently no support for Windows and no plan to add support. Please submit a PR if wish to add Windows support.

<a id='3a'></a>

### MATLAB Runtime Installation

For starting the manual installation, firstly the MATALB runtime engine must be installed as it is used for the FEA simulation within the environment. The version of the runtime that the package was built with is R2023a and can be downloaded from here: [MATLAB Runtime R2023a](https://ssd.mathworks.com/supportfiles/downloads/R2023a/Release/5/deployment_files/installer/complete/win64/MATLAB_Runtime_R2023a_Update_5_win64.zip).

Once you have downloaded the runtime, you can follow the instructions to install the runtime. The key part is to ensure that you retain the command shown below. This command is non-permanent and needs to be run every time a new terminal window is opened. Alternatively you can add this to your `~/.bashrc`.

```
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}\ 
  /path/to/install/dir/R2023a/runtime/glnxa64:\
  /path/to/install/dir/R2023a/bin/glnxa64:\
  /path/to/install/dir/R2023a/sys/os/glnxa64:\
  /path/to/install/dir/R2023a/extern/bin/glnxa64"
```

<a id='3b'></a>

### Package Installation

To install the necessary packages, it is recommended to use a virtual environment such as Ananconda or virtualenv. We will use Anaconda commands in this repo, but these could be substituted for other commands.

In the `manual` directory, use the provided `environment.yml` file to install all the necessary Python packages:

```
conda env create -f environment.yml
```

Once this is done, from the top level of this git repository, execute the following commands to install the wing panel FEA simulator package and the similar package for the wing spar FEA simulator:

```shell
# For the wing panel simulator
cd ./manual/train/calculateDeformationMARLTEST/for_redistribution_files_only
python3 setup.py install

# For the wing spar simulator
cd ./manual/train/calculateDeformationMARLSpar/for_redistribution_files_only
python3 setup.py install
```

The installation of the packages can be tested by running `import calculateDeformationMARL<type>` into a Python CLI instance, where `<type>` dictates either the panel (TEST) or spar (Spar) packages.

<a id='3c'></a>

### Training

For training the models, we provide a shell script that can be used to start a training process for both the spar and the panel models. The script must first be given executable privileges:

```shell
sudo chmod +x train_mafp.sh
```

To run the training cycle, execute the following command in the `/path/to/repo/manual/scripts/` folder:

```shell
./train_mafp.sh -e <env> -r <num runs> -n <num agents>
```

Where `env` is the environment to run (either panel or spar), `<num runs>` is the number of runs of the framework to run and `<num agents>` is the number of agents to use per run. These cannot be changed between runs. All hyperparameters are kept in the runner file. 

If you are using [Weights & Biases](https://wandb.ai/site) for logging, there are two optional arguments to pass that are required for logging:

```shell
./train_mafp.sh -e <env> -r <num runs> -n <num agents> -w -i <wandb identity>
```

Where `wandb identity` is your W&B account ID. Project name and run numbers are handled by the program config. All runs save the episodic reward, regret and step TD loss in CSV files, along with a network weights `.pt` file for each agent.

<a id='3d'></a>

### Inference

Once the agents have been trained, it is possible to run inference on the drilling positions. We provide a shell script to run inference for the agents. Similar to the training process, we first have to enable the script for execution:

```shell
sudo chmod +x eval_mafp.sh
```

Now you can run the script itself with the following parameters:

```shell
./eval_mafp.sh -e <env> -r <num runs> -n <num agents> -a <run name>
```

The flags `-e -r` and `-n` are identical to the flags in the train script. The main difference is the `-a` flag, which specifies which run you want to use to evaluation.

We provide networks weights for evaluation from our testing. They can be found in the `train/agent_weights` directory. To run the `eval` weights:

```shell
./eval_mafp.sh -e <env> -r <num runs> -n <num agents> -a eval
```

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
