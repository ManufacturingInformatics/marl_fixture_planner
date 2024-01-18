## Manual Installation

This work is built on two pillars: an FEA simulator developed in MATLAB and a multi-agent reinforcement learning process developed in Python. The system specs that the model was trained on is as follows:

```shell
OS: Ubuntu 20.04 LTS
Python Version: 3.10.10
GPU: NVIDIA GeForce RTX 3080
CPU: Intel Core i9-10920X 12C/24T
RAM: 64 GB
```

This work has only been tested on Ubuntu - there is currently no support for Windows and no plan to add support. Please submit a PR if wish to add Windows support.

### MATLAB Runtime Installation

For starting the manual installation, firstly the MATALB runtime engine must be installed as it is used for the FEA simulation within the environment. The instructions to install the runtime can be found [here](https://uk.mathworks.com/help/compiler/install-the-matlab-runtime.html). The version of the runtime that the package was built with is R2023a and can be downloaded from here: [MATLAB Runtime R2023a](https://ssd.mathworks.com/supportfiles/downloads/R2023a/Release/5/deployment_files/installer/complete/win64/MATLAB_Runtime_R2023a_Update_5_win64.zip).

Once you have downloaded the runtime, you can follow the instructions to install the runtime. The key part is to ensure that you retain the command shown below. This command is non-permanent and needs to be run every time a new terminal window is opened. Alternatively you can add this to your `~/.bashrc`.

```shell
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}\ 
  /path/to/install/dir/R2023a/runtime/glnxa64:\
  /path/to/install/dir/R2023a/bin/glnxa64:\
  /path/to/install/dir/R2023a/sys/os/glnxa64:\
  /path/to/install/dir/R2023a/extern/bin/glnxa64"
```

### Package Installation

To install the necessary packages, it is recommended to use a virtual environment such as Ananconda or virtualenv. We will use Anaconda commands in this repo, but these could be substituted for other commands.

In the `manual` directory, use the provided `environment.yml` file to install all the necessary Python packages:

```shell
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