FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

# ------ USER ROOT HAS BEEN ACTIVATED ------

# use root for package installation:
USER root

# create a non-root development user xdev:
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} xdev -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# ------ PART 0: set environment variables ------

# set up environment:
ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV HOME=/root SHELL=/bin/bash

# ------ PART 1: set up CN sources ------

# Ubuntu:
COPY ${PWD}/image/etc/apt/sources.list /etc/apt/sources.list
RUN rm -f /etc/apt/sources.list.d/*

# Python: 
COPY ${PWD}/image/etc/pip.conf /root/.pip/pip.conf 

# ------ PART 2: set up apt-fast -- NEED PROXY DUE TO UNSTABLE CN CONNECTION ------

# install:
RUN apt-get update -q --fix-missing && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
        # apt utils:
        apt-utils dpkg pkg-config \ 
        # PPA utilities:
        software-properties-common \
        # certificates management:
        dirmngr gnupg2 \
        # download utilities:
        axel aria2 && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-keys 1EE2FF37CA8DA16B && \
    add-apt-repository ppa:apt-fast/stable && \
    apt-get update -q --fix-missing && \
    apt-get install -y --no-install-recommends --allow-unauthenticated apt-fast && \
    rm -rf /var/lib/apt/lists/*

# CN config:
COPY ${PWD}/image/etc/apt-fast.conf /etc/apt-fast.conf

# ------ PART 3: add external repositories ------

# libsparse:
RUN add-apt-repository -r ppa:bzindovic/suitesparse-bugfix-1319687

# ------ PART 4: install packages ------

RUN apt-fast update --fix-missing && \
    apt-fast install -y --no-install-recommends --allow-unauthenticated \
        # security:
        openssh-server pwgen ca-certificates \
        # network utils:
        curl wget iputils-ping net-tools \
        # command line:
        vim grep sed patch \
        # io:
        rsync pv zip unzip bzip2 \
        # version control:
        git mercurial subversion \
        # daemon & services:
        supervisor nginx \
        # vnc support, potential image & rich text IO:
        lxde \
        xvfb dbus-x11 x11-utils libxext6 libsm6 x11vnc \
        gtk2-engines-pixbuf gtk2-engines-murrine pinta ttf-ubuntu-font-family \
        mesa-utils libgl1-mesa-dri libxrender1 \
        texlive-latex-extra \
        # c++:
        gcc g++ \
        make cmake build-essential autoconf automake libtool \
        libglib2.0-dev libboost-dev libboost-all-dev libtbb-dev \
        # python 3:
        ninja-build python3-dev python3-tk python3-opencv \
        # development common:
        lua5.3 liblua5.3-dev libluabind-dev \
        libgoogle-glog-dev \
        libsdl1.2-dev \
        libsdl-image1.2-dev \
        # numerical optimization:
        coinor-libcoinutils-dev \
        coinor-libcbc-dev \
        libeigen3-dev \
        gfortran \
        libopenblas-dev liblapack-dev \
        libdw-dev libatlas-base-dev libsuitesparse-dev \
        libmetis-dev \
        # graph optimization -- https://github.com/RainerKuemmerle/g2o
        # a. visualization:
        libqt4-dev libqt4-opengl-dev \
        qt5-default qt5-qmake qtdeclarative5-dev libqglviewer-dev-qt5 \
        # b. numerical optimization:
        libcholmod3 libcxsparse3 \
        # GUI tools:
        freeglut3-dev \
        gnuplot \
        gnome-themes-standard \
        terminator \
        firefox && \
    apt-fast autoclean && \
    apt-fast autoremove && \
    rm -rf /var/lib/apt/lists/*

# install pip
ENV PATH="/root/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py --user && rm get-pip.py

# ordered startup fix for supervisord:
RUN pip install ordered-startup-supervisord

# ------ PART 5: offline installers ------

# load installers:
COPY ${PWD}/installers /tmp/installers
WORKDIR /tmp/installers

# install tini:
RUN chmod u+x ./download-tini.sh && ./download-tini.sh && dpkg -i tini.deb && \
    apt-get clean

RUN rm -rf /tmp/installers

# ------ PART 6: set up VNC servers ------

COPY image /

WORKDIR /usr/lib/

RUN git clone https://github.com/novnc/noVNC.git -o noVNC

WORKDIR /usr/lib/noVNC/utils

RUN git clone https://github.com/novnc/websockify.git -o websockify

WORKDIR /usr/lib/webportal

# VNC server:
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 80 5901 9001

# ------ USER ROOT HAS BEEN DEACTIVATED ------

# ------- USER XDEV HAS BEEN ACTIVATED -------

# ------ PART 7: set up Anaconda environment ------

# installation path:
WORKDIR /opt

# install Anaconda in silent mode:
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda.sh && \
    chmod +x anaconda.sh && ./anaconda.sh -b -p anaconda

# add conda configs:
COPY ${PWD}/environment /opt/environment

# config conda:
RUN anaconda/bin/conda init && cp /opt/environment/.condarc /root/.condarc

#
# config environments:
#
# a. X-VectorNet development:

RUN anaconda/bin/conda env create -f /opt/environment/vectornet/gpu.yml

# here the installation of PyTorch Geometric is separated from conda env due to potential incompatibility:
ENV CUDA_HOME="/usr/local/cuda-11.0"

RUN /opt/anaconda/envs/graphnn/bin/pip install pytest-runner && \
    /opt/anaconda/envs/graphnn/bin/pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html && \
    /opt/anaconda/envs/graphnn/bin/pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html && \
    /opt/anaconda/envs/graphnn/bin/pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html && \
    /opt/anaconda/envs/graphnn/bin/pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html && \
    /opt/anaconda/envs/graphnn/bin/pip install torch-geometric

# b. reference benchmark system from FAIR:


# ------ USER XDEV HAS BEEN DEACTIVATED ------

# ------------------ DONE -----------------------

# enable dependency lib linking:
ENV LD_LIBRARY_PATH=/usr/local/lib

ENTRYPOINT ["/startup.sh"]