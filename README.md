<div align="center">
<img src=".assets/match_logo.png" width="700"/>
</div>

---
**MATCH** (**M**odel-**A**ware **T**VM-based **C**ompiler for **H**eterogeneous hardware) is a DNN compiler that exploits [Apache TVM](https://tvm.apache.org/)'s BYOC framework, targeting the optimized deployment of DNN applications onto heterogeneous edge System-on-Chip (SoC) platforms.

MATCH partitions a DNN isolating (or _matching_) patterns supported by each executional module in the target system. It then manages the compilation process for nodes supported by the available accelerators, and exploits the baseline TVM to generate code for unsupported nodes on the main host core. Compilation in MATCH is guided by a temporal mapping engine, which searches for the optimal schedule (i.e. loop ordering and memory allocation) of each layer. MATCH currently supports [ZigZag](https://github.com/KULeuven-MICAS/zigzag) as the default temporal mapping engine. At the end of the compilation process, MATCH produces C code to execute all nodes of the network, which users can invoke from their own "main" program to implement a complete inference.

Currently, MATCH supports the following targets:
- [DIANA](https://ieeexplore.ieee.org/document/9731716), a multi-accelerator SoC by KU Leuven.
- [GAP9](https://greenwaves-technologies.com/gap9_processor/), a commercial RISC-V-based parallel ultra-low-power platform.

Importantly, MATCH is designed to make the process of supporting new targets as simple as possible, thanks to a compilation process guided by _high-level model-based hardware abstractions_. Read below for instructions on this extension process. We plan to include several additional SoCs in the list of officially supported ones in the coming months.

## Reference

If you use MATCH, please acknowledge our paper:
```
@ARTICLE{10946988,

  author={Hamdi, Mohamed Amine and Daghero, Francesco and Sarda, Giuseppe Maria and Delm, Josse Van and Symons, Arne and Benini, Luca and Verhelst, Marian and Pagliari, Daniele Jahier and Burrello, Alessio},

  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 

  title={MATCH: Model-Aware TVM-Based Compilation for Heterogeneous Edge Devices}, 

  year={2025},

  volume={},

  number={},

  pages={1-1},

  keywords={Hardware;Artificial neural networks;Artificial intelligence;Memory management;Tensors;Design automation;Single instruction multiple data;Integrated circuit modeling;Computational modeling;Pattern matching;AI Compilers;Deep Neural Networks;Heterogeneous Computing;Deep Learning Accelerators},

  doi={10.1109/TCAD.2025.3556967}}

```

# Docker

The easiest way to use MATCH is through a Ubuntu22.04 docker container, which ensures all dependencies (especially for TVM) are met. This can be achieved with:
```
$ docker build -t match -f docker/Dockerfile .
$ docker start -it --rm match
```

# Local Installation
## Requirements
These instructions will consider a Ubuntu installation including:
- LLVM
- a C++ compiler with C++17 support, like GCC 7.1 or Clang 5.0
- CMake 3.18 or higher
- python3
- pip3

This can be achieved by running the following command on a fresh Ubuntu 22.04 install:

```
$ xargs -a system_requirements.txt sudo apt install -y
```
To install the latest release (with pip):

```
$ git clone --recursive https://github.com/eml-eda/match
$ cd match
$ python3 -m venv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
$ TVM_NCORES_INSTALL=$(nproc) make build_tvm
$ python3 setup.py install
```

Due to some environment dependencies it is reccomended to either set the correct environment running `source sourceme.sh` on the new terminal or by exporting directly on the user .bashrc the correct environment. 

# Usage

To use MATCH directly, the end user can execute the run.py script, setting the target, the type of the input and the input file.

Considering an ONNX network the user that should be compiled for a pulp platform, the user shall execute the following command

```
$ python3 test/test.py --model onnx image_classification --executor graph
```

<!--Additionally there are 2 predefined networks, that can be used to test a newly built target, that can be accessed from MATCH directly.
The first command issues a 2d requantized convolution with a bias add operation, while the second does an addition between 2 2d convolutions as the previous ones.-->

The user can also import match in a python script and compile a network defined in the TVM Relay IR language as follows:
```python
from match import match
from match.target.target import MatchTarget
mod,params = get_network_relay_ir()
target: MatchTarget = MyTarget()
match.match(
    model=MatchModel(
        relay_mod=mod, relay_params=params,
        model_name="default", executor="graph",
        default_inputs=default_inputs,
        golden_cpu_model=golden_cpu_model,
    ),
    target=target,
    output_path=output_path
)
```
This can be done also for ONNX files:
```python
from match import match
from match.target.target import MatchTarget
target: MatchTarget = MyTarget()
match.match(
    model=MatchModel(
        filename=onnx_model_filepath,
        model_type="onnx",
        model_name="default", executor="graph",
        default_inputs=default_inputs,
        golden_cpu_model=golden_cpu_model,
    ),
    target=target,
    output_path=output_path
)
```

<!--This API returns a result that contains the data about the compiled network. It is an instance of the `CompiledModuleResult`, this class saves the list of inputs with their types and sizes and the size of the output. Finally also the partitioned relay module is saved and can be used by the user to extract other informations.-->

# Extending MATCH to target new SoCs

To define a new target (i.e.,  a new heterogeneous SoC), users shall extend the `MatchTarget` class.
The extension process is quite simple, since the only thing that the user must provide is a list of _execution modules_, passed to the constructor of the base `MatchTarget` class.
Additionally with the host memory hierarchy definition and a set of paths and APIs.
Execution modules are classes that represent all hardware modules in the target SoC that MATCH should consider to implement the inference of DNN nodes, except for the main host CPU (which is handled by TVM). So, execution modules might include GPUs, dataflow accelerators, slave CPU clusters, etc.

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
        super(ExampleDigitalModule, self).__init__(name="digital",
                                                    libs_required={})
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
from match.partition.utils import add_checks_get_first_op

def conv2d_bias_add_pattern():
    # define pattern from the innermost operations up to the outermost one
    conv_2d = is_op("nn.conv2d")(
        wildcard(), wildcard()
    )
    return is_op("nn.bias_add")(conv_2d,wildcard())

def check_conv_2d_bias_add(node):
    bias_add = add_checks_get_first_op(node, "nn.bias_add")
    conv_2d = add_checks_get_first_op(node, "nn.conv2d")
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

conv_pattern_checks = PartitioningPattern(name="default_conv2d_bias_add", pattern=conv2d_bias_add_pattern additional_checks=check_conv_2d_bias_add)

dense_pattern = PartitioningPattern(name="default_conv2d_bias_add", pattern=dense_bnorm_pattern)
```

Therefore the exec module partitioning_function should look like:
```python
class ExampleDigitalModule(ExecModule):
    ...
    def partitioning_patterns(self):
        return [conv_pattern_checks, dense_pattern]
```
### Memory Hierarchy
MATCH also requires a definition of the _memory hierarchy_ of each execution module.
Noteworthy, this _memory hierarchy_ should contain only the specific memories of the executional module, without considering the ones already part of the host CPU. 
This is needed because MATCH considers the execution of a layer to happen inside the lowest memory-level. Thanks to ZigZag's temporal mapping engine, when the L1 is not large enough to entirely contain the inputs/outputs of a layer, the latter is tiled appropriately.

<!--So, each hierarchy definition should contain at least a 2 (or more) levels since the optimization stops there (TBC).-->

Hierarchies can be defined extending the `module_memories` method. A memory hierarchy is defined as a Python list of `MemoryInst` objects, ordered from lower level to higher level.

Each `MemoryInst` constructor receives the name of the memory, its size in kB, and the types of tensors(DNN weights, activations, or both).
Optionally, users can also specify double-buffering support, or indicate a size in bytes to be reserved for pattern-specific memory buffers.

Optionally the user can set also the number of read/write/rw ports and how they should be connected to the lower and higher memory level. The default implementation consists of a single read and write port on each side. More informations about the port settings can be seen in [Memory Docs](docs/dev/memory.md).

The default hierarchy implementation is a 2-level one, with a 32kB L1 and a 128kB L2, each accepting both weights and activations.

### Optimal Spatial Mapping
Users can also extend the definition of the optimal spatial mapping for an execution module.

The optimal spatial mapping is used to compute the _actual_ spatial mapping and so the number of virtual execution units that will be processing the layer in parallel. For a simple accelerator, optimal and actual spatial mapping are the same thing. However, for more complex platforms, such as multi-core CPU clusters, they might differ. Check the paper for more details.

Importantly, the optimal spatial mapping definition may be different for each supported pattern, to support accelerators that use different types of spatial unrolling for different DNN ops. An example of an optimal spatial mapping definition is shown below:
```python
def zigzag_optimal_spatial_mapping_def(self, match_node: MatchNode=None, pattern_name = "conv2d"):
    if pattern_name == "pointwise_conv2d":
            return [
                ("OY",8),("OX",2),("K",4)
            ]
    elif "dense" in pattern_name:
        return [
            ("K",8)
        ]
    else:
        return [
            ("OY",8),("OX",2),("K",4)
        ]
```

### Execution Cost Model
To guide the temporal mapping engine, the user shall provide a customized cost model, which computes the expected latency or energy of each candidate scheduling. This is done extending the `ZigZagMatchCostModel` class and 2 of its methods:
* `def_transfer_cost` that shall return the latency/energy cost of a data transfer given its size, the type of operand, and the involved memory levels.
* `def_innermost_loops_cost`, that computes the cost of a single call to the "innermost" execution loops (i.e., the back-end kernel for the given execution module, executing entirely from L1).

Importantly, `def_transfer_cost` is always called prior to `def_innermost_loops_cost`, so the user can set some parameters in the former, and use them later for the loops cost calculation.

For example, the following code defines the cost model of an accelerator, for which each transfer has an overhead of 100 cycles w.r.t. the cycles lost on the transfer itself, and where the number of cycles lost on the innemorst computations equals to 10000:
```python
from match.cost_model.zigzag import ZigZagMatchCostModel

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
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access,
            has_any_additional_buffer=True
        )
    
    def def_transfer_cost(self):
        return {operand:100 for operand in self.operands}
    
    def def_innermost_loops_cost(self):
        return 10000

class ExampleDigitalModule(ExecModule):
    ...
    def zigzag_cost_model(self):
        return ExampleCostModel
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
def comp_apis_def(self, computational_apis: ComputationalApis=None, pattern_name: str="conv2d"):
    comp_apis.compute_tile="example_kernel_function"
    return comp_apis
```

More detailed informations about the APIs can be seen in [Types,APIs and Libraries Docs](docs/dev/apis.md).

# License
MATCH entire codebase is released under [Apache License 2.0](LICENSE).
