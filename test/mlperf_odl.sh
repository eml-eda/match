#!/usr/bin/env bash
# KWS TESTS
# POPEN
# python3 test_odl.py --target pulp_open --relay_model models/odl/kws/model_graph_fw.relay --relay_params_filename models/odl/kws/model_params.txt --executor graph
# rm builds/baseline/kws/fw/popen -r
# mv builds/last_build builds/baseline/kws/fw/popen

# python3 test_odl.py --target pulp_open --relay_model models/odl/kws/model_graph_bw.relay --relay_params_filename models/odl/kws/model_params.txt --executor graph
# rm builds/baseline/kws/bw/popen -r
# mv builds/last_build builds/baseline/kws/bw/popen
# # GAP9
# python3 test_odl.py --target GAP9 --relay_model models/odl/kws/model_graph_fw.relay --relay_params_filename models/odl/kws/model_params.txt --executor graph
# rm builds/baseline/kws/fw/gap -r
# mv builds/last_build builds/baseline/kws/fw/gap

# python3 test_odl.py --target GAP9 --relay_model models/odl/kws/model_graph_bw.relay --relay_params_filename models/odl/kws/model_params.txt --executor graph
# rm builds/baseline/kws/bw/gap -r
# mv builds/last_build builds/baseline/kws/bw/gap

# VWW TESTS
# POPEN
# python3 test_odl.py --target pulp_open --relay_model models/odl/vww/model_graph_fw.relay --relay_params_filename models/odl/vww/model_params.txt --executor graph
# rm builds/baseline/vww/fw/popen -r
# mv builds/last_build builds/baseline/vww/fw/popen

# python3 test_odl.py --target pulp_open --relay_model models/odl/vww/model_graph_bw.relay --relay_params_filename models/odl/vww/model_params.txt --executor graph
# rm builds/baseline/vww/bw/popen -r
# mv builds/last_build builds/baseline/vww/bw/popen
# GAP9
python3 test_odl.py --target GAP9 --relay_model models/odl/vww/model_graph_fw.relay --relay_params_filename models/odl/vww/model_params.txt --executor graph
rm builds/baseline/vww/fw/gap -r
mv builds/last_build builds/baseline/vww/fw/gap

python3 test_odl.py --target GAP9 --relay_model models/odl/vww/model_graph_bw.relay --relay_params_filename models/odl/vww/model_params.txt --executor graph
rm builds/baseline/vww/bw/gap -r
mv builds/last_build builds/baseline/vww/bw/gap