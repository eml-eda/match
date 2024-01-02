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
- a C++ compiler with C++17 support, like GCC 7.1 or Clang 5.0
- CMake 3.18 or higher
- python3
- pip3

A fresh install of Ubuntu 22.04 should satify all requirements.

# Installation
To install the latest release (with pip):

```
$ git clone --recursive https://github.com/eml-eda/match
$ make all
```

When using a new fresh terminal, users can run `source sourceme.sh` to correctly set the environment. 

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

# Extending MATCH to target new SoCs
To use MATCH, a user can select among two main APIs: `with_relay` and `with_onnx`. The first one expects a DNN defined in the TVM Relay IR language, whereas the second one expects an ONNX model.

## Targets

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

Execution modules shall inherit from the `ExecModule` class. The latter contains a default implementation of all the functions needed by MATCH to target a new accelerator, allowing users to only customize the relevant ones.

### Supported Patterns
First of all, each `ExecModule` should define a list of supported DNN layer _patterns_, by extending the `partitioning_patterns` method.
A pattern in MATCH is a Relay IR expression corresopnding to a sequence of DNN operations. For instance, a given accelerator might match a sequence consisting of a convolution followed by cast, clip, and shift, to implement re-quantization.

Patterns themselves are defined through the `PartitioningPattern` class, whose parameters are:
* The pattern name, which should be unique within the target.
* A function returning the pattern in Relay
* A function that optionally checks for supplementary conditions that shall be true in order for the pattern to be supported by the target accelerator (e.g., a specific convolution kernel size).

An example of pattern definition is shown below:
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

### Memory Hierarchy
MATCH also requires a definition of the _memory hierarchy_ of each execution module. This is needed because MATCH considers the execution of a layer to happen inside the lowest memory-level. Thanks to ZigZag's temporal mapping engine, when the L1 is not large enough to entirely contain the inputs/outputs of a layer, the latter is tiled appropriately.

<!--So, each hierarchy definition should contain at least a 2 (or more) levels since the optimization stops there (TBC).-->

Hierarchies can be defined extending the `memories_def` method, and setting the `self.platform_mem` from therein. A memory hierarchy is defined as a Python list of `MemoryInst` objects, ordered from lower level to higher level.

Each `MemoryInst` constructor receives the name of the memory, its size in kB, and the types of operand supported (DNN weights, activations, or both). Optionally, users can also specify double-buffering support, or indicate a size in bytes to be reserved for pattern-specific memory buffers.

For example, the following code defines a memory with ....(TBD):
```python
ADD EXAMPLE
```

The default hierarchy implementation is a 2-level one, with a 32kB L1 and a 128kB L2, each accepting both weights and activations.

### Optimal Spatial Mapping
Users can also extend the definition of the optimal spatial mapping for an execution module, by setting the `self.optimal_spatial_mapping` attribute within the function `optimal_spatial_mapping_def`.

The optimal spatial mapping is used to compute the _actual_ spatial mapping and so the number of virtual execution units that will be processing the layer in parallel. For a simple accelerator, optimal and actual spatial mapping are the same thing. However, for more complex platforms, such as multi-core CPU clusters, they might differ. Check the paper for more details.

Importantly, the optimal spatial mapping definition may be different for each supported pattern, to support accelerators that use different types of spatial unrolling for different DNN ops.An example of an optimal spatial mapping definition is shown below:
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

### Execution Cost Model
To guide the temporal mapping engine, the user shall provide a customized cost model, which computes the expected latency or energy of each candidate scheduling. This is done extending the `ZigZagMatchCostModel` class and 2 of its methods:
* `def_transfer_cost` that shall return the latency/energy cost of a data transfer given its size, the type of operand, and the involved memory levels.
* `def_innermost_loops_cost`, that computes the cost of a single call to the "innermost" execution loops (i.e., the back-end kernel for the given execution module, executing entirely from L1).

Importantly, `def_transfer_cost` is always called prior to `def_innermost_loops_cost`, so the user can set some parameters in the former, and use them later for the loops cost calculation.

For example, the following code defines ....(TBD):
```python
ADD EXAMPLE
```


### C Backend APIs
Lastly, to complete the definition of an execution module, the user must define the names of the C backend functions that MATCH should invoke to use the accelerators. The functions that MATCH requires are organized in flexible templates, some of which can optionally be empty for accelerators that do not require a certain step (e.g., an initialization prior to starting an inference). In detail, MATCH APis are divided into:
* Memory functions
* Computation functions
* Synchronization functions
* Platform setup functions
* Type-definitions (for defining some specific types used by MATCH, such as the kernel structure used).

For example, the following code defines ....(TBD):
```C
ADD EXAMPLE
```

Once a new target has been defined, users shall add it to the targets list within the file `match/target/get_target.py` in order to inform MATCH of its existence. Then, users can compile for the new target as shown in the Usage section above.

# License
MATCH entire codebase is released under [Apache License 2.0](LICENSE).
