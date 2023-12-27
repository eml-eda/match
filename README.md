# MATCH
MATCH is a DNN compiler that exploits the TVM BYOC framework to extend the capacity of the former tool.

This project targets users who may want to deploy optimally a DNN application onto a heterogeneous edge platform.

MATCH partitions a network with patterns for a specific module of the target or the host core(a normal CPU). With a partioned network MATCH manages then the compilation process of the layers supported by the targeted platform, and exploits TVM for the compilation process of the layers assigned to the host core.

The compilation in MATCH is guided by a temporal mapping engine, which searches for the optimal schedule(i.e. loop ordering and memory allocation). MATCH currently supports ZigZag as the temporal mapping engine.

Finally MATCH will output all the generated code, for each fused layer separetely

# Target definition example
To use MATCH a user can use one of the defined APIs. The main 2 ones are; with_relay and with_onnx. The first one expects a network and its parameters expressed in the TVM Relay IR language, and the name of the target to use. The second one instead expects an ONNX model and the name of target used.

To define a new target the user may extend a class, MatchTarget.
The extension process is quite simple since the only thing that the user must provide is the list of execution module in the __init__ to the original MatchTarget class. For example to define a target which contains a digital accelerator and an analog one a user must declare a class as following:
```python
from match.target.target import MatchTarget

class ExampleTarget(MatchTarget):
    def __init__(self):
        super(ExampleTarget,self).__init__([
            ExampleAnalogModule(),
            ExampleDigitalModule(),
        ],name="example")
```

The execution modules are classes that represent each component that will be used for the inference of a DNN fused layer. So it may represent a CPU, GPU, a dedicated accelerator, a cluster of CPUs or any computational component.

Also for their definition the user must extend a Match class. This one is ExecModule, which contains the definition of all the necessary information of a module, with a default implementation to allow the user to customize only the relevant ones.

Each ExecModule should support a list of patterns, a pattern in MATCH defines a fused layer, where there should be a computational layer and supporting ones, such as cast,clip,shift and many others. An ExecModule defines the supported patterns by extending the partitioning_patterns function.

Patterns are defined through the PartitioningPattern class, that contains the name of the pattern(which should be unique throughout the target, and will be used for the rest of the process), the pattern itself, which is a function that returns a pattern defined in the TVM Relay IR language and finally a function that receives the matched patterns and issues all the remaining necessary conditions to support it. An example of a pattern can be shown below:
```python
from match.partition.partitioning_pattern import PartitioningPattern

def conv2d_pattern():
    return is_op("nn.conv2d")(
        wildcard(), wildcard()
    )

def no_checks(pattern):
    return True

pattern = PartitioningPattern(name="default_conv2d",pattern=conv2d_pattern,additional_checks=no_checks)
```

MATCH considers each module separately, containing each an independent memory hierarchy, that can be defined setting self.platform_mem extending the memories_def class function. The default implementation is a 2 level memory hierarchy of shared memories, the lower level one of 32kB and the higher level one of 128kB.

MATCH considers the execution of a fused layer to happen inside of the lowest memory level defined, so each instance should contain at least a 2(or more) level memory hierarchy, since the optimization stops there.

A memory hierarchy is defined as a Python list of MemoryInst classes. The list should be ordered from lower level memories to higher level ones.

MemoryInst defines a memory instance and contains the name of the memory, its size in kB, the operands that are targeted by this memory and many more data that can be customized as preferred. These vary from the double buffering support to a function that computes the number of bytes that are allocated on the memory for buffers used for a specific pattern.

The user is also advised to extend the definition of the optimal spatial mapping, by setting self.optimal_spatial_mapping in the function optimal_spatial_mapping_def. The optimal spatial mapping is used to compute the actual spatial mapping and so the number of virtual units that should be executing the layer. This definition may depend on the pattern, an example of an optimal spatial mapping definition can be found below:
```python
def optimal_spatial_mapping_def(self, pattern_name: str = "default_conv2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
    if pattern_name=='default_conv2d' and (dim_sizes['FY']*dim_sizes['FX'])==1:
        # pointwise conv
        self.optimal_spatial_mapping = [
            ("OY",4),("OX",4),("K",4)
        ]
    else:
        self.optimal_spatial_mapping = [
            ("K",16),("OY",4)
        ]
```

To guide the temporal mapping engine research the user may deploy a customized cost model, which computes the expected latency and energy of each solution. The default temporal mapping engine used currently is ZigZag so it is encouraged to extend the ZigZagMatchCostModel class and 2 of its methods; def_transfer_cost that shall return the transfer cost of each operand for every memory transfer iteration and def_innermost_loops_cost, that instead computes the cost of a single call to the innermost loops(back-end kernels). def_transfer_cost is called previously with respect to def_innermost_loops_cost so the user can set some parameters for the class earlier and use them later for the loops cost calculation.

Finally to complete the definition of an executional model the user must define the names of the apis used by MATCH, that are divided into memory related ones, computational ones, synchronization ones, platform ones and also the definition of some specific types used by MATCH, such as the kernel structure used.

With a defined target the user currently to extend MATCH has to add the target to the target list in match/target/get_target.py .
# Requirements
These instructions will consider a Ubuntu installation that satify these constraints:
- a C++ compiler with C++17 support, like GCC 7.1 or Clang 5.0
- CMake 3.18 or higher
- python3
- pip3

A fresh install of Ubuntu 22.04 should satify all this requirements beforehand.

# Installation
To install the latest release (with pip):

```
$ git clone --recursive https://github.com/eml-eda/match
$ make all
```

When using a new fresh terminal the user may have to run "source sourceme.sh" on top of the directory to set correctly the environment. 

# Usage

Considering a target the user may try 2 simple networks with some predefined apis, to do so the user can run one of the following commands:

```
$ python3 match/api.py -c -t target_name
$ python3 match/api.py -a -t target_name
```

Where the first command issues a 2d requantized convolution with a bias add operation, while the second does an addition between 2 2d convolutions as the previous ones.

The user can also import match in a python script and define a network in the TVM Relay IR language as follows:
```python
import match.api.with_relay as match_relay
mod,params=define_network()
match_relay(mod,params,target_name)
```
# License
MATCH entire codebase is released under [Apache License 2.0](LICENSE).