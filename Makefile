checkout:
	git submodule update --recursive --init

build_tvm:
	sudo apt-get update
	sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
	cd match-tvm; mkdir build; cp cmake/config.cmake build; cd build; cmake ..;make -j 4
	pip3 install --user typing-extensions psutil scipy numpy decorator attrs pybind11
	cd match-tvm; export TVM_HOME=$(CURDIR)
	export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

build_zigzag:
	pip3 install numpy networkx sympy matplotlib onnx tqdm multiprocessing_on_dill

build_match:
	python3 setup.py install --user

build: build_tvm build_zigzag build_match

all: checkout build