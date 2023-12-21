export TVM_HOME=$PWD/match-tvm
echo "Set TVM_HOME as" $TVM_HOME
export PYTHONPATH=$TVM_HOME/python:$PWD/zigzag:${PYTHONPATH}
echo "Updated PYTHONPATH as" $PYTHONPATH