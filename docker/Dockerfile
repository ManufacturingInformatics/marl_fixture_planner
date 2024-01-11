FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

SHELL ["/bin/bash", "-c"]

COPY ./environment.yml /opt/environment.yml

####################################################
# General System Updates 
####################################################

RUN apt update \
    && apt upgrade -y \
    && apt install -y wget unzip \
    && rm -rf /var/lib/apt/lists/*

####################################################
# Python Installation & Package Installation
####################################################

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && chmod +x Miniconda3-latest-Linux-x86_64.sh \
    && ./Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda update -y --name base conda && conda clean --all -y
RUN conda --version

RUN conda env create -f /opt/environment.yml
ENV PATH /opt/conda/envs/rl-fixture-planner/bin:$PATH

RUN echo "conda activate rl-fixture-planner" >> ~/.bashrc

# TODO: Find a way to get the conda env to activate and then stay activated

####################################################
# MATLAB Runtime Installation
####################################################

RUN wget \
    https://ssd.mathworks.com/supportfiles/downloads/R2023a/Release/5/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2023a_Update_5_glnxa64.zip \
    && unzip -d /home/matlab_runtime MATLAB_Runtime_R2023a_Update_5_glnxa64.zip
    
WORKDIR "/home/matlab_runtime"
RUN chmod +x install \
    && ./install -agreeToLicense yes -destinationFolder /usr/MATLAB/MATLAB_Runtime -outputFile runtime_log.txt

RUN export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}\
    /usr/MATLAB/MATLAB_Runtime/R2023a/runtime/glnxa64:\
    /usr/MATLAB/MATLAB_Runtime/R2023a/bin/glnxa64:\
    /usr/MATLAB/MATLAB_Runtime/R2023a/sys/os/glnxa64:\
    /usr/MATLAB/MATLAB_Runtime/R2023a/extern/bin/glnxa64"

# TODO: Finalise the last pieces of code for the docker file
# TODO: Test the docker run commands for training and inference
