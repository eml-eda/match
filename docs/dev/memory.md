# Memory

In MATCH the user has to define, for each MatchTarget, a memory hierarchy, this is defined through an ordered list, from the lower level memories up to the higher level ones, of memory instances.

# Memory instance

A memory instance is defined through the MemoryInst class, this class gives the user the possibility to define a memory, setting the size in kBs, the number of read, write and read/write ports.

Optionally the user can set the double buffering support, in case of lower level memories for example.

<!--Finally the user can also set directly how this memory will be connected to the other ones with the `used_ports` parameter-->
The default implementation is comprised of a single read/write port for the higher level memory and the lower level one, the user can anyway set the ports independently.

To define a memory instance with 48kB the user shall define an object as below:

```python
MemoryInst(name="example_mem",k_bytes=48)
```