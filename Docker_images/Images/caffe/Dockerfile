# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

# Ubuntu 18.04 (bionic) from 2018-04-26
# https://github.com/docker-library/official-images/commit/aac6a45b9eb2bffb8102353c350d341a410fb169
FROM ubuntu:bionic-20180426@sha256:c8c275751219dadad8fa56b3ac41ca6cb22219ff117ca98fe82b42f24e1ba64e

LABEL maintainer="Jupyter Scipybase"

USER root

# Install all OS dependencies for notebook server that starts but lacks all
# features (e.g., download as all possible file formats)
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -yq dist-upgrade \
 && apt-get install -yq --no-install-recommends \
    build-essential \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    libncurses5-dev \
    vim \
    git \
    jed \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    pandoc \
    python-dev \
    unzip \
    nano \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Install Tini
RUN wget --quiet https://github.com/krallin/tini/releases/download/v0.18.0/tini && \
    echo "12d20136605531b09a2c2dac02ccee85e1b874eb322ef6baf7561cd93f93c855 *tini" | sha256sum -c - && \
    mv tini /usr/local/bin/tini && \
    chmod +x /usr/local/bin/tini

# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home

USER root

RUN mkdir -p $CONDA_DIR && \ 
# Setup work directory for backward-compatibility
    mkdir /home/work 

# Install conda and check the md5 sum provided on the download site
ENV MINICONDA_VERSION 4.5.1
RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "0c28787e3126238df24c5d4858bd0744 *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
    $CONDA_DIR/bin/conda install --quiet --yes conda=4.5.4 && \
    $CONDA_DIR/bin/conda update --all --quiet --yes && \
    conda clean -tipsy && \
    rm -rf /home/.cache/yarn 

# Install Jupyter Notebook and Hub
#RUN conda install --quiet --yes \
RUN conda install --yes \
    'notebook=5.5.*' \
    'jupyterlab=0.32.*'  \
    'nodejs=9.11*'  && \
    conda clean -tipsy && \
    jupyter labextension install @jupyterlab/hub-extension@^0.8.1 && \
    npm cache clean --force && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    rm -rf /home/.cache/yarn 

USER root
#EXPOSE 8888
WORKDIR $HOME/work

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]

# Add local files as late as possible to avoid cache busting
RUN wget -P /usr/local/bin/  https://raw.githubusercontent.com/jupyter/docker-stacks/master/base-notebook/start.sh && \
    wget -P /usr/local/bin/  https://raw.githubusercontent.com/jupyter/docker-stacks/master/base-notebook/start-notebook.sh && \
    wget -P /usr/local/bin/  https://raw.githubusercontent.com/jupyter/docker-stacks/master/base-notebook/start-singleuser.sh && \
    wget -P /etc/jupyter/    https://raw.githubusercontent.com/jupyter/docker-stacks/master/base-notebook/jupyter_notebook_config.py

# ffmpeg for matplotlib anim
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends ffmpeg && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

# Install pytorch-caffe2
# You can choose: pytorch-caffe2-cuda9.0-cudnn7,  pytorch-caffe2-cuda8.0-cudnn7 and more info. in https://anaconda.org/caffe2/repo
#RUN conda install --quiet --yes \
#
RUN conda install --yes \
    'qt=5.9.*' \
    'pyqt=5.9.*'

RUN conda install --yes  -c caffe2 pytorch-caffe2 && \
    conda install --yes  torchvision && \
    conda remove --quiet --yes --force qt pyqt && \
    conda clean -tipsy && \
    # Activate ipywidgets extension in the environment that runs the notebook server
    #jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
    # Also activate ipywidgets extension for JupyterLab
    #jupyter labextension install @jupyter-widgets/jupyterlab-manager@^0.35 && \
    #jupyter labextension install jupyterlab_bokeh@^0.5.0 && \
    
    npm cache clean --force && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    rm -rf /home/.cache/yarn && \
    rm -rf /home/.node-gyp 

# Install facets which does not have a pip or conda package at the moment
#RUN cd /tmp && \
#    git clone https://github.com/PAIR-code/facets.git && \
#    cd facets && \
#    #jupyter nbextension install facets-dist/ --sys-prefix && \
#    cd && \
#    rm -rf /tmp/facets 

# Import matplotlib the first time to build the font cache.
#ENV XDG_CACHE_HOME /home/.cache/
#RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" 
