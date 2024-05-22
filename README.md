<div align="center">
<img src=".assets/match_logo.png" width="700"/>
</div>

---
**MATCH** (**M**odel-**A**ware **T**VM-based **C**ompiler for **H**eterogeneous hardware) is a DNN compiler that exploits [Apache TVM](https://tvm.apache.org/)'s BYOC framework, targeting the optimized deployment of DNN applications onto heterogeneous edge System-on-Chip (SoC) platforms.

MATCH partitions a DNN isolating (or _matching_) patterns supported by each hardware module in the target system. It then manages the compilation process for layers supported by the available accelerators, and exploits the baseline TVM to generate code for unsupported layers on the main host core. Compilation in MATCH is guided by a temporal mapping engine, which searches for the optimal schedule (i.e. loop ordering and memory allocation) of each layer. MATCH currently supports [ZigZag](https://github.com/KULeuven-MICAS/zigzag) as the default temporal mapping engine. At the end of the compilation process, MATCH produces C code to execute all layers of the network, which users can invoke from their own "main" program to implement a complete inference.

Currently, MATCH supports the following targets:
- [DIANA](https://ieeexplore.ieee.org/document/9731716), a multi-accelerator SoC by KU Leuven.
- [GAP9](https://greenwaves-technologies.com/gap9_processor/), a commercial RISC-V-based parallel ultra-low-power platform.

Importantly, MATCH is designed to make the process of supporting new targets as simple as possible, thanks to a compilation process guided by _high-level model-based hardware abstractions_. Read below for instructions on this extension process. We plan to include several additional SoCs in the list of officially supported ones in the coming months.

## Reference

If you use MATCH, please acknowledge our paper: `UNDER REVIEW`
```
UNDER REVIEW!!!
```

# Requirements
These instructions will consider a Ubuntu installation including:
- LLVM
- a C++ compiler with C++17 support, like GCC 7.1 or Clang 5.0
- CMake 3.18 or higher
- python3
- pip3

This can be achieved with

```
$ sudo apt install -y llvm cmake python3 python3-pip 
```

A fresh install of Ubuntu 22.04 should satify all requirements.

# Installation
To install the latest release (with pip):

```
$ git clone --recursive https://github.com/eml-eda/match
$ cd match
$ make all
```

When using a new fresh terminal, users can run `source sourceme.sh` to correctly set the environment. 

# Usage

To use MATCH directly, the end user can execute the run.py script, setting the target, the type of the input and the input file.

Considering an ONNX network the user that should be compiled for gap9, the user shall execute the following command

```
$ python3 match/run.py -t target_name -i onnx -f examples/small_mobilenet_V1.onnx
```

<!--Additionally there are 2 predefined networks, that can be used to test a newly built target, that can be accessed from MATCH directly.
The first command issues a 2d requantized convolution with a bias add operation, while the second does an addition between 2 2d convolutions as the previous ones.-->

The user can also import match in a python script and compile a network defined in the TVM Relay IR language as follows:
```python
from match import match
mod,params=define_network()
res=match(relay_mod=mod,relay_params=params,target_name="default")
```

This API returns a result that contains the data about the compiled network. It is an instance of the `CompiledModuleResult`, this class saves the list of inputs with their types and sizes and the size of the output. Finally also the partitioned relay module is saved and can be used by the user to extract other informations.

This API accepts also directly an instance of the target if it is not defined directly inside of the codebase:
```python
from match import match
mod,params=define_network()
target=ExampleTarget()
match(relay_mod=mod,relay_params=params,target=target)
```

If the network is instead exposed in a file, the user has to set the input type(i.e. onnx, relay etc.) and the path for the file containing the module (for the relay case the user shall also pass the file containing the parameters).

```python
from match import match
target=ExampleTarget()
match(input_type="onnx",filename="examples/small_mobilenet_V1.onnx",target=target)
```

```python
from match import match
target=ExampleTarget()
match(input_type="relay",filename="examples/quant_conv.relay",params_filename="examples/params_quant_conv.txt",target=target)
```
# Extending MATCH to target new SoCs

To define a new target (i.e.,  a new heterogeneous SoC), users shall extend the `MatchTarget` class.
The extension process is quite simple, since the only thing that the user must provide is a list of _execution modules_, passed to the constructor of the base `MatchTarget` class. Execution modules are classes that represent all hardware modules in the target SoC that MATCH should consider to implement the inference of DNN layers, except for the main host CPU (which is handled by TVM). So, execution modules might include GPUs, dataflow accelerators, slave CPU clusters, etc.

For example to define a target SoC which contains a digital accelerator and an analog one, a user could use the following code:
```python
from match.target.target import MatchTarget

class ExampleTarget(MatchTarget):
    def __init__(self):
        super(ExampleTarget,self).__init__([
            ExampleAnalogModule(),
            ExampleDigitalModule(),
        ],name="example")
```

## Execution Modules

Each execution module essentially defines a **model-based hardware abstraction** of a specific accelerator.

Execution modules shall inherit from the `ExecModule` class. The latter contains a default for all the pieces of information needed by MATCH to support a new accelerator, allowing users to only customize the relevant ones. 

```python
from typing import Dict
from match.target.exec_module import ExecModule

class ExampleDigitalModule(ExecModule):
    def __init__(self):
        super(ExampleDigitalModule, self).__init__(name="cluster",
                                          src_path="./src",
                                          inc_path="./include")
```

The main items that require customization are the following.

### Supported Patterns
First of all, each `ExecModule` should define a list of supported DNN layer _patterns_, by extending the `partitioning_patterns` method.
A pattern in MATCH is a Relay IR expression corresponding to a sequence of DNN operations. For instance, a given accelerator might match a sequence consisting of a convolution followed by cast, clip, and shift, to implement re-quantization.

Patterns are defined through the `PartitioningPattern` class, whose parameters are:
* The pattern name, which should be unique within the target.
* A function returning the pattern in Relay
* The relay name of the pattern operation that the MATCH framework will use as a base for the loop ordering aspect of scheduling.
* An optional function that checks for supplementary conditions that shall be true in order for the pattern to be supported by the target accelerator (e.g., a specific convolution kernel size).

In the example below there are defined 2 patterns, one with a conv 2d and a bias add operation that requires the conv2d to have a balanced kernel shape, and a dense(fully connected) fused with a batch norm operation(multiply and add):
```python
from match.partition.partitioning_pattern import PartitioningPattern

def conv2d_bias_add_pattern():
    # define pattern from the innermost operations up to the outermost one
    conv_2d = is_op("nn.conv2d")(
        wildcard(), wildcard()
    )
    return is_op("nn.bias_add")(conv_2d,wildcard())

def check_conv_2d_bias_add(pattern):
    bias_add=pattern
    conv_2d=bias_add.args[0]
    # if this conv has an unbalanced kernel return False, this pattern is not supported
    if conv_2d.attrs["kernel_size"][0]!=conv_2d.attrs["kernel_size"][1]:
        return False
    # pattern supported
    return True

def dense_bnorm_pattern():
    dense = is_op("nn.dense")(
        wildcard(), wildcard()
    )
    mult = is_op("multiply")(dense,wildcard())
    return is_op("add")(mult,wildcard())

conv_pattern_checks = PartitioningPattern(name="default_conv2d_bias_add",pattern=conv2d_bias_add_pattern,ordered_operation="nn.conv2d",additional_checks=check_conv_2d_bias_add)

dense_pattern = PartitioningPattern(name="default_conv2d_bias_add",pattern=dense_bnorm_pattern,ordered_operation="nn.dense")
```

Therefore the exec module partitioning_function should look like:
```python
class ExampleDigitalModule(ExecModule):
    ...
    def partitioning_patterns(self):
        return [conv_pattern_checks, dense_pattern]
```
### Memory Hierarchy
MATCH also requires a definition of the _memory hierarchy_ of each execution module. This is needed because MATCH considers the execution of a layer to happen inside the lowest memory-level. Thanks to ZigZag's temporal mapping engine, when the L1 is not large enough to entirely contain the inputs/outputs of a layer, the latter is tiled appropriately.

<!--So, each hierarchy definition should contain at least a 2 (or more) levels since the optimization stops there (TBC).-->

Hierarchies can be defined extending the `memories_def` method. A memory hierarchy is defined as a Python list of `MemoryInst` objects, ordered from lower level to higher level.

Each `MemoryInst` constructor receives the name of the memory, its size in kB, and the types of operand supported (DNN weights, activations, or both). Optionally, users can also specify double-buffering support, or indicate a size in bytes to be reserved for pattern-specific memory buffers. To achieve this the user shall define a function, that takes as an input the informations of the accelerated layer and the name of the pattern that we're matching. Therefore, this function should adhere to the following prototype:
```python
def buffer_func(layer_data,pattern_name):
    return 0
```

For example, the following code defines a memory instance with 64kB that is used for the input and the output, which supports double buffering, that keeps reserved 4 bytes for each output channel("K") in case of a convolution:
```python
def memory_buffer(layer_data,pattern_name):
    if pattern_name=="default_conv2d_bias_add":
        return layer_data.loop_dim_size["K"]*4
    else:
        return 0
MemoryInst(name="example_mem",k_bytes=64,operands=["I","O"],double_buffering_support=True,buffer_for_layer_func=memory_buffer)
```

Optionally the user can set also the number of read/write/rw ports and how they should be connected to the lower and higher memory level. The default implementation consists of a single read and write port on each side. More informations about the port settings can be seen in [Memory Docs](docs/dev/memory.md).

Finally the extended method will look like this:

```python
def memories_def(self, pattern_name, operands):
    return [MemoryInst(name="example_l1_mem",k_bytes=64,operands=["I","O","W"],double_buffering_support=True,buffer_for_layer_func=memory_buffer),MemoryInst(name="example_l2_mem",k_bytes=256,operands=["I","O","W"])]
```
The default hierarchy implementation is a 2-level one, with a 32kB L1 and a 128kB L2, each accepting both weights and activations.

### Optimal Spatial Mapping
Users can also extend the definition of the optimal spatial mapping for an execution module.

The optimal spatial mapping is used to compute the _actual_ spatial mapping and so the number of virtual execution units that will be processing the layer in parallel. For a simple accelerator, optimal and actual spatial mapping are the same thing. However, for more complex platforms, such as multi-core CPU clusters, they might differ. Check the paper for more details.

Importantly, the optimal spatial mapping definition may be different for each supported pattern, to support accelerators that use different types of spatial unrolling for different DNN ops. An example of an optimal spatial mapping definition is shown below:
```python
def optimal_spatial_mapping_def(self, pattern_name: str = "",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
    if pattern_name=='default_conv2d' and (dim_sizes['FY']*dim_sizes['FX'])==1:
        # pointwise conv
        return [
            ("OY",4),("OX",4),("K",4)
        ]
    else:
        return [
            ("K",16),("OY",4)
        ]
```

### Execution Cost Model
To guide the temporal mapping engine, the user shall provide a customized cost model, which computes the expected latency or energy of each candidate scheduling. This is done extending the `ZigZagMatchCostModel` class and 2 of its methods:
* `def_transfer_cost` that shall return the latency/energy cost of a data transfer given its size, the type of operand, and the involved memory levels.
* `def_innermost_loops_cost`, that computes the cost of a single call to the "innermost" execution loops (i.e., the back-end kernel for the given execution module, executing entirely from L1).

Importantly, `def_transfer_cost` is always called prior to `def_innermost_loops_cost`, so the user can set some parameters in the former, and use them later for the loops cost calculation.

For example, the following code defines the cost model of an accelerator, for which each transfer has an overhead of 100 cycles w.r.t. the cycles lost on the transfer itself, and where the number of cycles lost on the innemorst computations equals to the nunmber of output channels("K"):
```python
from match.target.cost_model import ZigZagMatchCostModel
from math import prod,ceil,floor

class ExampleCostModel(ZigZagMatchCostModel):
    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
    ):
        super(ExampleCostModel,self).__init__(
            accelerator=accelerator,layer=layer,spatial_mapping=spatial_mapping,
            temporal_mapping=temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access)
    
    def def_transfer_cost(self):
        return {operand:input_transfer_costs[operand][0]+100 if operand!="O" else self.self.output_transfer_costs[0]+100 for operand in self.operands}
    
    def def_innermost_loops_cost(self):
        return self.loop_sizes['K']
```

More detailed examples can be found in [Cost Model Docs](docs/dev/cost_model.md)

### C Backend APIs
Lastly, to complete the definition of an execution module, the user must define the names of the C backend functions that MATCH should invoke to use the accelerator. The functions that MATCH requires are organized in flexible templates, some of which can optionally be empty for accelerators that do not require a certain step (e.g., an initialization prior to starting an inference). In detail, MATCH APis are divided into:
* Memory functions
* Computation functions
* Synchronization functions
* Platform setup functions
* Type-definitions (for defining some specific types used by MATCH, such as the kernel structure used).

For example, the following code defines the computational APIs extending the `comp_apis_def` function of the executional model, which receives the default computational apis object and returns the updated one:
```python
def comp_apis_def(self,comp_apis: ComputationalApis=ComputationalApis()):
    comp_apis.innermost_computation="example_kernel_function"
    return comp_apis
```

More detailed informations about the APIs can be seen in [Types,APIs and Libraries Docs](docs/dev/apis.md).

Once a new target has been defined, to use it directly within MATCH, users shall add it to the targets list within the file `match/target/get_target.py` in order to inform MATCH of its existence. Otherwise the user can use the target through the API. Then, users can compile for the new target as shown in the Usage section above.

# License
MATCH entire codebase is released under [Apache License 2.0](LICENSE).
