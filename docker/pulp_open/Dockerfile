ARG TVM_NCORES_INSTALL=1
# Starting from Ubuntu 22.04
FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]
# Setting values for tzdata
ENV DEBIAN_FRONTEND=noninteractive
# General
RUN apt update
# utility
RUN apt install -y vim git wget
# pulp-sdk system dependencies
RUN apt install -y build-essential libftdi-dev libftdi1 doxygen python3-pip libsdl2-dev curl cmake \
    libusb-1.0-0-dev scons gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev
# pulp-sdk python dependencies
RUN pip3 install argcomplete pyelftools
# Change directory to /
WORKDIR /

RUN wget https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/download/v1.0.16/v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2

RUN tar -xf v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2

RUN rm v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2

RUN mv /v1.0.16-pulp-riscv-gcc-ubuntu-18 /pulp_toolchain

WORKDIR /pulp_toolchain

# Make the sourceme.sh file executable
RUN chmod +x sourceme.sh

# Source the sourceme.sh file
RUN /bin/bash -c "source sourceme.sh"

WORKDIR /

RUN git clone --recursive https://github.com/pulp-platform/pulp-sdk.git

WORKDIR /pulp-sdk

RUN git submodule update --remote --recursive

RUN rm -r tools/gvsoc

WORKDIR /pulp-sdk/tools

RUN git clone --recursive https://github.com/eml-eda/gvsoc.git

WORKDIR /pulp-sdk

RUN pip3 install argcomplete pyelftools prettytable six mako rich

ENV PULP_RISCV_GCC_TOOLCHAIN=/pulp_toolchain

RUN /bin/bash -c "source configs/pulp-open.sh"

RUN chmod +x configs/pulp-open.sh

# build pulp-sdk
RUN pip3 install -r /pulp-sdk/tools/gvsoc/requirements.txt
RUN pip3 install -r /pulp-sdk/tools/gvsoc/gapy/requirements.txt
RUN pip3 install -r /pulp-sdk/tools/gvsoc/core/requirements.txt
RUN cd /pulp_toolchain; source sourceme.sh; cd /pulp-sdk; source configs/pulp-open.sh; make build -j 4
RUN echo "source /pulp-sdk/configs/pulp-open.sh" >> ~/.bashrc

WORKDIR /

RUN git clone --recursive https://github.com/eml-eda/match.git

WORKDIR /match

RUN git submodule update --init --recursive

RUN xargs -a system_requirements.txt apt install -y

RUN pip3 install -r requirements.txt

RUN TVM_NCORES_INSTALL=${TVM_NCORES_INSTALL} make build_tvm_single_core

RUN python3 setup.py install

ENV TVM_HOME="/match/match-tvm"
ENV PYTHONPATH="${TVM_HOME}/python:/match/zigzag:"
