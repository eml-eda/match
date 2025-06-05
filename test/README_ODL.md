# On-device Learning on PULP-Open Cluster

## Install
TODO

## Getting Started
1. Clone PULP-TrainLib 

cd test/targets/modules/libs/
git clone https://github.com/pulp-platform/pulp-trainlib ./pulp-trainlib/
git checkout 4dcce609bffcf2544efba298fb62281147f09b65


2. Run Test Example
a) Forward model
python test/test_odl.py --target pulp_open --relay_model $PWD//test/models/odl/test_model/model_graph_fw.relay --relay_params_filename $PWD/test/models/odl/test_model/model_params.txt --executor graph

b) Backward Model
python test/test_odl.py --target pulp_open --relay_model $PWD/test/models/odl/test_model/model_graph_bw.relay --relay_params_filename $PWD/test/models/odl/test_model/model_params.txt --executor graph


3. 
after generating forward or backward code:
cd test/builds/last_build
make clean all run