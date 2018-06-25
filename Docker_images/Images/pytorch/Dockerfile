# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
FROM jupyter/scipy-base:v1 

LABEL maintainer="Jupyter Pytorch"

USER root


# pytorch  
# please vist the URL to check more verions for cpu or gpu
RUN conda install --quiet --yes  -c pytorch pytorch && \
    conda install --quiet --yes  torchvision &&\
    conda remove --quiet --yes --force qt pyqt && \
    conda clean -tipsy && \
    npm cache clean --force && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    rm -rf /home/.cache/yarn && \
    rm -rf /home/.node-gyp 
