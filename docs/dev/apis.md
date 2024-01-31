# Types

MATCH objective resides in extensibility, so to fulfill this point it leaves the possibility to the user to define for the code generation phase some types and structures instead of using the default ones.

Currently, the user can extend the kernel structure, to add maybe pointers to internal buffers or save other key information that is not yet captured by MATCH. The other extendible type is instead for the declaration of variables and constants in MATCH, in fact some platforms to save a constant in the higher level memories require using special macros and types.

The user can then extend the `types_def` method and set new types as it is shown below:

```python
def types_def(self,match_types: MatchTypes=MatchTypes()):
    match_types.kernel_struct="example_kernel"
    match_types.mem_data_macro_and_type="L2_DATA uint8_t"
    return match_types
```

# APIs

On the MATCH framework there are a set of APIs that are used during the code generation phase to obtain a functional program, these functions can be reimplemented by the user, who should adhere to the original prototypes, and inform MATCH of the name of these APIs.

In almost all APIs a pointer to a `common_kernel` structure is passed, this is to give more freedom to the user, since this structure contains informations about the current pattern.

## Memory APIs

The memory APIs are the main ones, they contain all the ones used to obtain the offsets for the pointers, up to the ones referred to actual memory transfers.

To allow users to allocate and setup the necessary memory MATCH exposes an API for it, this API has the following prototype:
```C
void match_startup_memory(common_kernel* common_kernel,int* first_op_sizes,unsigned int first_op_db,void* dim_first,
                                int* second_op_sizes,unsigned int second_op_db,void* dim_sec,
                                int* third_op_sizes,unsigned int third_op_db,void* dim_third,
                                int* paddings,int* strides);
```

While, to eventually deallocate the memory, or just to shut it down the API prototype is instead the following:
```C
void match_shutdown_mem(common_kernel* common_kernel);
```

MATCH provides also a set of APIs to compute the offsets given the tile indexes, currently there is a default implementation for every data layout. This is the list of APIs that MATCH provides:

```C
unsigned int match_pointer_offset_layout_O(common_kernel* common_kernel,tile_indexes_O* tile_idxs,unsigned int memory_level);
unsigned int match_pointer_offset_layout_I(common_kernel* common_kernel,tile_indexes_I* tile_idxs,unsigned int memory_level);
unsigned int match_pointer_offset_layout_X(common_kernel* common_kernel,tile_indexes_X* tile_idxs,unsigned int memory_level);
unsigned int match_pointer_offset_layout_Y(common_kernel* common_kernel,tile_indexes_Y* tile_idxs,unsigned int memory_level);
```

The user should reimplement a general API for these purposes if the weights are not saved totally contigously, but maybe are interleaved with constants.

Regarding the constants, the code generation outputs a call before a general synchronization to an API that should be responsible to load constants that may be needed during computation(biases etc.):

```C
void match_pattern_constants_loading(match_kernel* kernel,unsigned int iter,void* weights_and_constant_buf);
```

Finally the last set of APIs are the ones that refers to the actual memory transfers. Also for these APIs there is one defined for each type of operand:

```C
unsigned int match_mem_transfer_I(common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem);
unsigned int match_mem_transfer_X(common_kernel* common_kernel,dimension_X* dim,unsigned int ext_pt,int ext_mem,int int_mem);
unsigned int match_mem_transfer_Y(common_kernel* common_kernel,dimension_Y* dim,unsigned int ext_pt,int ext_mem,int int_mem);
unsigned int match_mem_transfer_W(common_kernel* common_kernel,dimension_W* dim,unsigned int ext_pt,int ext_mem,int int_mem);
```

Having reimplemented one or more of these APIs the user shall inform MATCH about their names by extending the `mem_apis_def` method. This method receives a default instance of `MemoryApis`. To use `example_pattern_constants_loading` instead of the default implementation the method will look like:

```python
def mem_apis_def(self,memory_apis: MemoryApis=MemoryApis()):
    memory_apis.pattern_constants_loading="example_pattern_constants_loading"
    return memory_apis
```

## Computational APIs

Following, thecomputational APIs, refer to the ones that are tied directly to computation and to the kernel structure itself.
Currently there are 2 APIs, one that sets the customized parameter of the personalized kernel structure, and one that is used to do computation for the current tile indexes.

The first one has the following prototype:

```C
void match_init_other_kernel_params(example_kernel* kernel);
```

`match_kernel` is the default kernel structure used, that just contains a pointer to the common kernel structure. To use a different and more complex kernel structure the user has to extend the `types_def` function and set the correct name.
The latter API instead has the following prototype:

```C
void match_innermost_computation(match_kernel* kernel);
```

To set the reimplemented APIs in MATCH the user has to extend the `comp_apis_def` method. This method receives a default instance of `ComputationalApis`. To use `example_innermost_computation` instead of the default implementation and also `example_set_kernel` the method will look like:

```python
def comp_apis_def(self,computational_apis: ComputationalApis=ComputationalApis()):
    computational_apis.innermost_computation="example_innermost_computation"
    computational_apis.init_other_kernel_params="example_set_kernel"
    return computational_apis
```

## Sync APIs

Regarding instead the synchronizational aspect, MATCH currently utilizes 3 APIs; the first of which is used to wait for a set of asynchronous transfers. The second is instead used in regards to the computational aspect, to wait for the end of the past computation (this can be used to exploit the double buffering mechanism). Lastly MATCH has an API to wait for the current transfer to finish instead.

Here are the prototypes:

```C
void match_async_transfers(common_kernel* common_kernel);

void match_prev_computation(common_kernel* common_kernel);

void match_curr_computation(common_kernel* common_kernel);
```

The method to extend in this case is the `sync_apis_def` method. This method receives a default instance of `SyncApis`. To use `wait_transfers` instead of the default implementation the method will look like:

```python
def sync_apis_def(self,sync_apis: SyncApis=SyncApis()):
    sync_apis.async_transfers="wait_transfers"
    return sync_apis
```

## Platform APIs

Finally the last set of APIs regard the executional platform. Currently there is only an API that initialize the platform(it could be used to configure a cluster for example).

This API has the following prototype:

```C
void match_init_platform(void (inner_function)(unsigned int* args_inner_function),unsigned int* args);
```

This API takes as an input the function that implements all the pattern and its argument, the default implementation just calls this function, so it should be enough if there is not any special requirement.

The method to extend in this case is the `platform_apis_def` method. This method receives a default instance of `PlatformApis`. To use `example_init_platform` instead of the default implementation the method will look like:

```python
def platform_apis_def(self,platform_apis: PlatformApis=PlatformApis()):
    platfomr_apis.init_platform="example_init_platform"
    return platform_apis
```

# Libraries

The code generation process inserts into the files a list of libraries, this list can be incremented extending the `def_include_list` function, for example to include in the files the library "example_mem.h" the function will look like:

```python
def def_include_list(self):
    return ["example_mem.h"] 
```