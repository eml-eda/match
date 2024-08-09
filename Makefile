checkout:
	git submodule update --recursive --init

sources:
	chmod 777 ./sourceme.sh

build_tvm:
	sudo apt-get update
	sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
	cd match-tvm; mkdir build; cp cmake/config.cmake build; cd build; cmake ..;make -j 8
	pip3 install --user typing-extensions psutil scipy numpy decorator attrs pybind11

build_zigzag:
	pip3 install --user numpy networkx sympy matplotlib onnx tqdm multiprocessing_on_dill

build_match:
	python3 setup.py install --user

build: build_tvm build_zigzag build_match

all: sources checkout build