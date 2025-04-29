ARG TVM_NCORES_INSTALL=1
# Starting from Ubuntu 22.04
FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]
# Setting values for tzdata
ENV DEBIAN_FRONTEND=noninteractive
# General
RUN apt update
# utility
RUN apt install -y vim git

WORKDIR /

COPY . /match

WORKDIR /match

RUN xargs -a system_requirements.txt apt install -y

RUN pip3 install -r requirements.txt

RUN git submodule update --init --recursive

RUN TVM_NCORES_INSTALL=${TVM_NCORES_INSTALL} make build_tvm_single_core

RUN python3 setup.py install --user

ENV TVM_HOME="/match/match-tvm"
ENV PYTHONPATH="${TVM_HOME}/python:/match/zigzag:"
