# Memory

In MATCH the user has to define, for each executional model, a memory hierarchy, this is defined through an ordered list, from the lower level memories up to the higher level ones, of memory instances.

# Memory instance

A memory instance is defined through the MemoryInst class, this class gives the user the possibility to define a memory, setting the size in kBs, the number of read, write and read/write ports.

Optionally the user can set the double buffering support, in case of lower level memories for example.

Finally the user can also define a function that given the pattern can subtract some bytes for some buffers used in the implementation of the operation. This function, takes as an input the informations of the accelerated layer and the name of the pattern that we're matching. Therefore, this function should adhere to the following prototype:

```python
def buffer_func(layer_data,pattern_name):
    return 0
```

<!--Finally the user can also set directly how this memory will be connected to the other ones with the `used_ports` parameter-->
The default implementation is comprised of a single read/write port for the higher level memory and the lower level one, the user can anyway set the ports independently.

To define a memory instance that has only a read port and a specific write port with 48kB the user shall define an object as below:

```python
MemoryInst(name="example_mem",k_bytes=48,operands=["I","O"],double_buffering_support=True,r_ports=1,w_ports=1,rw_ports=0)
```