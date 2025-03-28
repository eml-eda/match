ARG TVM_NCORES_INSTALL=8
# Starting from Ubuntu 22.04
FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]
# Setting values for tzdata
ENV DEBIAN_FRONTEND=noninteractive
# General
RUN apt update
# utility
RUN apt install -y vim git

COPY . /match

WORKDIR /match

RUN git submodule update --init --recursive

RUN xargs -a system_requirements.txt apt install -y

RUN pip3 install -r requirements.txt

RUN make build_tvm -j${TVM_NCORES_INSTALL}

RUN python3 setup.py install

ENV TVM_HOME="/match/match-tvm"
ENV PYTHONPATH="${TVM_HOME}/python:/match/zigzag:"
