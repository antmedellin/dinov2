# build command  docker build -t dino2_docker .  

# FROM nvidia/cuda:12.5.0-devel-ubuntu22.04
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y

RUN apt install -y \
    git \
    build-essential \
    wget \
    unzip \
    pkg-config \
    cmake \
    pip \
    sudo \
    g++ \
    ca-certificates


RUN pip install \
    torch \
    torchvision\
    omegaconf \
    torchmetrics==0.10.3 \
    fvcore \
    iopath \
    xformers==0.0.18 \
    submitit 
    # cuml-cu11
    
RUN pip install \
    matplotlib \
    ipykernel \
    opencv-python \
    scikit-learn \
    albumentations \
    transformers \
    evaluate

RUN apt install -y \
    libgl1-mesa-glx \
    gdal-bin
    
RUN pip install \
    lightning \
    tensorboard \
    torch-tb-profiler \
    pandas \
    matplotlib \
    seaborn 

RUN pip install --upgrade torchmetrics 

RUN apt-get install -y python3-tk

RUN pip install segmentation-models-pytorch 

RUN useradd -m dino_user 

RUN echo "dino_user ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/dino_user

USER dino_user

WORKDIR /home