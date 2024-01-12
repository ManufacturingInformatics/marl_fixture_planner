FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

SHELL ["/bin/bash", "-c"]

COPY ./environment.yml /opt/environment.yml
ADD ./manual/train/calculateDeformationMARLTEST /opt/calculateDeformationMARLTEST
ADD ./manual/train/calculateDeformationMARLSpar /opt/calculateDeformationMARLSpar

RUN mkdir /home/code
ADD /manual /home/code/manual

####################################################
# General System Updates 
####################################################

RUN apt update \
    && apt upgrade -y \
    && apt install -y wget unzip libxrender1 libxtst6 libxi6 default-jre\
    && rm -rf /var/lib/apt/lists/*

####################################################
# MATLAB Runtime Installation
####################################################

RUN wget \
    https://ssd.mathworks.com/supportfiles/downloads/R2023a/Release/5/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2023a_Update_5_glnxa64.zip \
    && unzip -d /home/matlab_runtime MATLAB_Runtime_R2023a_Update_5_glnxa64.zip \
    && rm -rf MATLAB_Runtime_R2023a_Update_5_glnxa64.zip
    
WORKDIR "/home/matlab_runtime"
RUN chmod +x install \
    && ./install -agreeToLicense yes -destinationFolder /usr/local/MATLAB/MATLAB_Runtime -outputFile runtime_log.txt

####################################################
# Python Installation
####################################################

ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

RUN conda update -y --name base conda && conda clean --all -y

RUN conda env create -f /opt/environment.yml
ENV PATH /opt/conda/envs/rl-fixture-planner/bin:$PATH

RUN echo "source activate /opt/conda/envs/rl-fixture-planner && export PYTHONPATH=$PYTHONPATH:/home/code" >> ~/.bashrc
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/MATLAB/MATLAB_Runtime/R2023a/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2023a/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2023a/sys/os/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2023a/extern/bin/glnxa64"
RUN source ~/.bashrc

####################################################
# FEA Package Installation
####################################################

WORKDIR /opt/calculateDeformationMARLTEST/for_redistribution_files_only
RUN python3 setup.py install

WORKDIR /opt/calculateDeformationMARLSpar/for_redistribution_files_only
RUN python3 setup.py install

####################################################
# On Container Startup
####################################################

WORKDIR /