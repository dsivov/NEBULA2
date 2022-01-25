FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

WORKDIR /app

COPY . .

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "source activate base" > ~/.bashrc

RUN apt-get update && apt-get install -y \
    python-dev \
    python-pip \
    gcc

SHELL ["/bin/bash", "-c"]
ARG CONDA_ENV=playground
RUN conda --version \ 
    && conda env create -n ${CONDA_ENV} -f ./env_playground.yml \
    && source activate ${CONDA_ENV}

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]


 



