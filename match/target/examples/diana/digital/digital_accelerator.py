import copy
import functools
from math import ceil, prod
import operator
import sys
from typing import Any, Dict, List
import numpy as np
from match.target.examples.diana.digital.cost_model import DigitalAcceleratorCostModel
from match.target.examples.diana.digital.network_transformations import network_transformations as diana_digital_net_trans
from match.target.examples.diana.digital.network_transformations import adjust_network as diana_digital_adj_net
from match.target.examples.diana.digital.partitioning_patterns import partitioning_patterns as diana_digital_patterns
from match.target.exec_module import ExecModule, PlatformApis, MemoryApis, SyncApis, ComputationalApis, MatchTypes
import os
from match.target.memory_inst import MemoryInst
import tvm

def bytaze(value,layer_name):
    if layer_name=='dense':
        byteval=None
        for ind in range(len(value)):
            valindval=int(value[ind])
            if valindval<0:
                valindval+=2**128
            valind=valindval.to_bytes(16,sys.byteorder)
            if ind==0:
                byteval=valind
            else:
                byteval+=valind
        return np.frombuffer(byteval,dtype='uint8')
    return np.frombuffer(value.tobytes(),dtype='uint8')

def channelize(value):
    pass

def depthwise_conv_2d_bias(value,match_node):
    k_size=match_node.const_tensors['W'].data.shape[0]
    npad = ((0, (16 - (value.shape[0] % 16)) % 16))
    value = np.pad(value, pad_width=npad, mode='constant', constant_values=0)
    value=value.reshape(value.shape[0],1)
    npad=((0,0),(0,3))
    value=np.pad(value, pad_width=npad, mode='constant', constant_values=0)
    value=bytaze(value,'depthwise_conv_2d')
    new_weights = []
    for pos in range(4):
        for ch in range(int(np.asarray(value).shape[0]/16)):
            for pos_in in [3,2,1,0]:
                new_weights.append(value[pos+4*(ch*4+pos_in)])
    final_weights = []
    for ch in range(int((k_size * 4 +15)/16)):
        for byte in range(4):
            final_weights.append(new_weights[(k_size * 4 * byte + ch*16):(k_size * 4 * byte + ch*16 + 16)])
    channels=ceil(np.asarray(final_weights).flatten().shape[0]/(16*4))
    return np.asarray(final_weights).flatten().reshape(channels,ceil(np.asarray(final_weights).flatten().shape[0]/channels)).tolist()

def dense_bias(value,match_node):
    npad = ((0, (16 - (value.shape[0] % 16)) % 16))
    value=np.pad(value, pad_width=npad, mode='constant', constant_values=0)
    initial_value=copy.deepcopy(value)
    value=bytaze(value,'dense')
    new_weights = []
    bias_bytes=8
    temp = copy.deepcopy(value)
    temp = temp.reshape(ceil(value.shape[0]/4), 4)
    temp1 = copy.deepcopy(temp)
    temp[:,0] = temp1[:,3] 
    temp[:,1] = temp1[:,2] 
    temp[:,2] = temp1[:,1] 
    temp[:,3] = temp1[:,0]
    temp=temp.flatten()
    final_weights=[]
    for ch in range(ceil(initial_value.shape[0]/16)):
        final_weights.append(temp[ch*256:(ch+1)*256].tolist())
    return final_weights

def conv_2d_bias(value,match_node):
    npad = ((0, (16 - (value.shape[0] % 16)) % 16))
    value=np.pad(value, pad_width=npad, mode='constant', constant_values=0)
    initial_value=copy.deepcopy(value)
    value=bytaze(value,'conv_2d')
    new_weights = []
    bias_bytes=4
    bias_bytes_ch=bias_bytes if value.shape[0]>bias_bytes else 1
    for pos in range(bias_bytes_ch):
        for ch in range(ceil(value.shape[0]/16)):
            for pos_in in [_ for _ in range(bias_bytes)][::-1]:
                new_weights.append(value[pos+bias_bytes_ch*(ch*bias_bytes+pos_in)])
    final_weights = []
    out_tensors_names = [n for n in match_node.output_tensors.keys()]
    out_dims = [d.size for d in match_node.output_tensors[out_tensors_names[0]].dims]
    channels=out_dims[1]
    for ch in range(ceil(channels/16)):
        for byte in range(bias_bytes_ch):
            final_weights.append(new_weights[(channels*byte + ch*16):(channels*byte + ch*16 + 16)])
    return np.asarray(final_weights).flatten().reshape(ceil(value.shape[0]/64),64 if value.shape[0]>64 else value.shape[0]).tolist()

def c_arr_shape(self,ref_type):
    arr=[val for val in ref_type.shape]
    size=0
    if len(arr)>0:
        size=functools.reduce(operator.mul,arr,1)
        arr=f'[{size}]'
    else:
        arr=''
    return arr,int(size)

def c_friendly_npvalue(arr):
    # params: arr is expected to be a numpy version of the value, it should be an array but it may be also just a single value
    if len(arr.shape)>0:
        # this is actually an array and not a single value
        arr=arr.reshape([arr.shape[0]]).astype(np.uint8)
        return f'{{{str(list(arr))[1:len(str(list(arr)))-1]}}}'
    else:
        return str(arr)

def adjust_value(self,ref):
    shape,size=self.c_arr_shape(ref.checked_type)
    if isinstance(ref,tvm.relay.Constant):
        return {'type':self.get_type(ref.checked_type.dtype),'shape':shape,'size':size,'value':self.c_friendly_npvalue(ref.data.numpy(),size)}
    else:
        return {'type':self.get_type(ref.checked_type.dtype),'shape':shape,'size':size}

def depthwise_conv_2d_weights(value,match_node):
    fy_fx=prod(match_node.ops["conv2d"].kernel_size)
    npad = ((0, (16 - (value.shape[0] % 16)) % 16), (0, 0), (0, 0), (0,0))
    value = np.pad(value, pad_width=npad, mode='constant', constant_values=0)
    for ch in np.arange(value.shape[0]):
        if ch == 0:
            temp = np.concatenate((value[ch,:,:,:].reshape(1,value.shape[1],value.shape[2],value.shape[3]), np.zeros((3, 1, value.shape[2], value.shape[3]))), axis = 0)
        else:
            temp1 = np.concatenate((value[ch,:,:,:].reshape(1,value.shape[1],value.shape[2],value.shape[3]), np.zeros((3, 1, value.shape[2], value.shape[3]))), axis = 0)
            temp = np.concatenate((temp, temp1), axis=0)
    temp = np.transpose(temp, (1, 2, 3, 0))
    temp = temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2],int(temp.shape[3]/16), 16)
    temp = np.transpose(temp, (3, 0, 1, 2, 4))
    flatvalue=temp.flatten()
    tempf = temp.reshape(ceil(flatvalue.shape[0]/4), 4)
    temp1 = copy.deepcopy(tempf)
    tempf[:,0] = temp1[:,3] 
    tempf[:,1] = temp1[:,2] 
    tempf[:,2] = temp1[:,1] 
    tempf[:,3] = temp1[:,0]
    channels=ceil(flatvalue.shape[0]/(16*fy_fx))
    return tempf.flatten().reshape(channels,ceil(flatvalue.shape[0]/channels)).tolist()

def conv_2d_weights(value,match_node):
    npad = ((0, (16 - (value.shape[0] % 16)) % 16), (0, 0), (0, 0), (0,0))
    temp = np.pad(value, pad_width=npad, mode='constant', constant_values=0)
    temp = np.transpose(temp, (1, 2, 3, 0))
    temp = temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2],ceil(temp.shape[3]/16), 16 if temp.shape[3]>16 else temp.shape[3])
    temp = np.transpose(temp, (3, 0, 1, 2, 4))
    flatvalue=temp.flatten()
    tempf = temp.reshape(ceil(flatvalue.shape[0]/4), 4)
    temp1 = copy.deepcopy(tempf)
    tempf[:,0] = temp1[:,3] 
    tempf[:,1] = temp1[:,2] 
    tempf[:,2] = temp1[:,1] 
    tempf[:,3] = temp1[:,0]
    return tempf.flatten().reshape(ceil(value.shape[0]/16),ceil(flatvalue.shape[0]/ceil(value.shape[0]/16))).tolist()

def dense_weights(value,match_node):
    npad = ((0, (16 - (value.shape[0] % 16)) % 16), (0, (16 - (value.shape[1] % 16)) % 16))
    value=np.pad(value, pad_width=npad, mode='constant', constant_values=0)
    temp = np.transpose(value)
    temp = temp.reshape(temp.shape[0],ceil(temp.shape[1]/16),16 if temp.shape[1]>16 else temp.shape[1])
    temp = np.transpose(temp,(2,0,1))
    flatvalue=value.flatten()
    tempf = value.reshape(ceil(flatvalue.shape[0]/4), 4)
    temp1 = copy.deepcopy(tempf)
    tempf[:,0] = temp1[:,3] 
    tempf[:,1] = temp1[:,2] 
    tempf[:,2] = temp1[:,1] 
    tempf[:,3] = temp1[:,0]
    return tempf.flatten().reshape(ceil(value.shape[0]/16),ceil(flatvalue.shape[0]/ceil(value.shape[0]/16))).tolist()


WEIGHT_ROUTER={'conv_2d':conv_2d_weights,'depthwise_conv_2d':depthwise_conv_2d_weights,'dense':dense_weights,'default':conv_2d_weights}

BIAS_ROUTER={'conv_2d':conv_2d_bias,'depthwise_conv_2d':depthwise_conv_2d_bias,'dense':dense_bias,'default':conv_2d_bias}

LAYERS_WITH_SAME_NAME=['depthwise']

def transform(key,value,match_node,layer_name):
    if ''.join([n for n in layer_name.split('_') if n not in LAYERS_WITH_SAME_NAME]) in key:
        return (WEIGHT_ROUTER[layer_name if layer_name in WEIGHT_ROUTER else 'default'](value.data.numpy(),match_node),key)
    elif 'bias' in key:
        return (BIAS_ROUTER[layer_name if layer_name in BIAS_ROUTER else 'default'](value.data.numpy(),match_node),key)
    else:
        return (None,(key,c_friendly_npvalue(value.data.numpy())))

def transform_key_name(key: str):
    if 'shift' in key:
        return 'output_shift'
    else:
        return key



class DigitalAccelerator(ExecModule):
    def __init__(self,**kwargs):
        super(DigitalAccelerator, self).__init__(name="digital",
                                          specific_patterns=[
                                              "conv_2d",
                                              "dense",
                                              "depthwise_conv_2d",
                                              "elem_add",
                                          ],
                                          src_path=os.path.dirname(__file__)+"/src",
                                          inc_path=os.path.dirname(__file__)+"/include",
                                          **kwargs)
        self.L1_SIZE=256 if "l1_size" not in kwargs else kwargs["l1_size"]

    def zigzag_optimal_spatial_mapping_def(self, match_node=None, pattern_name = "conv_2d"):
        return [
                ("K",16),("OX",16),
            ]
    
    def specific_pattern_def(self, match_node=None, pattern_name = "conv_2d"):
        if pattern_name=="conv2d" and match_node.ops["conv2d"].depthwise:
            return "depthwise_conv_2d"
        else:
            return pattern_name

    def memories_def(self, pattern_name, operands):
        mem = [
            # from lower level to higher level memories
            MemoryInst(name="act_mem",k_bytes=self.L1_SIZE,operands=[op for op in operands if op!="W"],r_bw=64*8,w_bw=64*8,r_ports=0,w_ports=0,rw_ports=2,double_buffering_support=False),
            MemoryInst(name="dram",k_bytes=512,operands=operands,r_ports=1,w_ports=1,rw_ports=0,r_bw=16*8,w_bw=16*8),
        ]
        if "W" in operands:
            mem = [
                MemoryInst(name="weight_mem",k_bytes=64,operands=["W"],double_buffering_support=False,r_bw=256 * 8, w_bw=128,r_ports=1,w_ports=1,rw_ports=0),
            ] + mem
        return mem
    
    def partitioning_patterns(self):
        return diana_digital_patterns()

    def network_transformations(self,opts):
        return diana_digital_net_trans(opts=opts)

    def adjust_network(self, opts):
        return diana_digital_adj_net(opts=opts)

    def def_include_list(self,patter_name):
        return ["digital_lib.h"]

    def mem_apis_def(self,mem_apis: MemoryApis=MemoryApis()):
        mem_apis.copy_out_curr_computation="digital_memcopyresult"
        mem_apis.mem_transfer={
            "I":"digital_memcopy_I",
            "X":"digital_memcopy_X",
            "Y":"digital_memcopy_Y",
            "W":"digital_memcopy_W",
            "O":"digital_memcopy_O",
        }
        mem_apis.pointer_offset["W"]="digital_W_pointer_offset"
        mem_apis.shutdown_mem="digital_free_channel"
        mem_apis.startup_memory="digital_set_channel"
        return mem_apis

    def comp_apis_def(self,comp_apis: ComputationalApis=ComputationalApis()):
        comp_apis.innermost_computation="diana_digital_kernel_wrapper"
        comp_apis.specific_pattern=self.specific_pattern
        return comp_apis

    def types_def(self,types: MatchTypes=MatchTypes()):
        types.mem_data_macro_and_type="L2_DATA uint8_t"
        return types

    def zigzag_cost_model(self):
        return DigitalAcceleratorCostModel
    
    #def layout_per_operand_def(self, pattern_name, specific_pattern, operands):
    #    return {operand:"NHWC" for operand in operands}
    
    #def adjust_network(self, opts):
    #    return gap_adjust_net(opts=opts)
    
    def weights_and_constants(self, match_node, pattern_name):
        """define how the weights and constants of a layer must be saved in C on the generated code

        Args:
            layer_arguments (List, optional): Dict of the arguments(parameters) for the node. Defaults to [].
        """
        out_tensors_names = [n for n in match_node.output_tensors.keys()]
        out_dims = [d.size for d in match_node.output_tensors[out_tensors_names[0]].dims]
        out_ch = out_dims[1]
        output_channels=out_ch if 'depthwise' not in pattern_name else out_ch*4
        constants=list()
        single_constants=dict()
        for (layer_arg_name,layer_arg_val) in match_node.const_tensors.items():
            if isinstance(layer_arg_val, tvm.relay.Constant):
                if len(layer_arg_val.data.shape)==0:
                    single_constants[layer_arg_name]=str(layer_arg_val.data)
                else:
                    constants.append(transform(layer_arg_name,layer_arg_val,match_node=match_node,pattern_name=pattern_name))
        diana_weights=[]
        for ch in range(ceil(output_channels/16)):
            for (c,keyval) in constants:
                if c is not None:
                    diana_weights+=c[ch]
                # elif c is None and ch==0:
                #     layer_data.layer_attrs[transform_key_name(keyval[0])]=keyval[1]
        diana_weights=(np.asarray(diana_weights).flatten())
        return {
            'value':c_friendly_npvalue(diana_weights),
            'len':int(diana_weights.shape[0]),
            'shape':f'[{ceil(diana_weights.shape[0])}]',
            "single_costants":single_constants,
        }
    def zigzag_architecture(self, optimal_spatial_mapping = None, platform_memories = None, match_node = None):
        from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
        from zigzag.classes.hardware.architecture.operational_unit import Multiplier
        from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
        from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
        from zigzag.classes.hardware.architecture.accelerator import Accelerator
        from zigzag.classes.hardware.architecture.core import Core
        def get_memory_hierarchy(multiplier_array,no_of_inputs):
            """Memory hierarchy variables"""
            """ size=#bit, bw=#bit"""
            # Defintion of register file for inputs
            if no_of_inputs>1:
                rf_1B_I_X = MemoryInstance(
                    name="rf_1B",
                    mem_type="rf",
                    size=32,
                    r_bw=32,
                    w_bw=32,
                    r_cost=1,
                    w_cost=1.2,
                    area=0,
                    r_port=1,
                    w_port=1,
                    rw_port=0,
                )
                rf_1B_I_Y = MemoryInstance(
                    name="rf_1B",
                    mem_type="rf",
                    size=32,
                    r_bw=32,
                    w_bw=32,
                    r_cost=1,
                    w_cost=1.2,
                    area=0,
                    r_port=1,
                    w_port=1,
                    rw_port=0,
                )
            else:
                rf_1B_I = MemoryInstance(
                    name="rf_1B",
                    mem_type="rf",
                    size=32,
                    r_bw=32,
                    w_bw=32,
                    r_cost=1,
                    w_cost=1.2,
                    area=0,
                    r_port=1,
                    w_port=1,
                    rw_port=0,
                )
                # Defintion of register file for weights
                rf_1B_W = MemoryInstance(
                    name="rf_1B",
                    mem_type="rf",
                    size=32,
                    r_bw=32,
                    w_bw=32,
                    r_cost=1,
                    w_cost=1.2,
                    area=0,
                    r_port=1,
                    w_port=1,
                    rw_port=0,
                )
                # Defintion of first SRAM for weights
                l1_w = MemoryInstance(
                    name="weight_mem",
                    mem_type="sram",
                    size=64 * 1024 * 8,
                    r_bw=256 * 8, w_bw=128,
                    r_cost=50,
                    w_cost=55,
                    area=0,
                    r_port=1,
                    w_port=1,
                    rw_port=0,
                )
            # Defintion of rRegister file for outputs
            rf_4B = MemoryInstance(
                name="rf_4B",
                size=32,
                r_bw=32,
                w_bw=32,
                r_cost=3,
                w_cost=3.6,
                area=0,
                r_port=2,
                w_port=2,
                rw_port=0,
            )

            shared_l1 = MemoryInstance(
                name="act_mem",
                size=self.L1_SIZE* 1024 * 8,
                r_bw=64 * 8,
                w_bw=64 * 8,
                r_cost=33.2 * 8,
                w_cost=38.5 * 8,
                area=0,
                r_port=0,
                w_port=0,
                rw_port=2,
                latency=1,
                min_r_granularity=32,
                min_w_granularity=32,
            )


            l2 = MemoryInstance(
                name="dram",
                size=512 * 1024 * 8 ,  # Size of L2 memory
                r_bw=16 * 8,
                w_bw=16 * 8,
                r_cost=100,
                w_cost=110,
                area=0,
                r_port=1,
                w_port=1,
                rw_port=0,
                latency=1,
            )  # rd E per bit 16

            memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)

            """
            fh: from high = wr_in_by_high = 
            fl: from low = wr_in_by_low 
            th: to high = rd_out_to_high = 
            tl: to low = rd_out_to_low = 
            """
            if no_of_inputs>1:
                memory_hierarchy_graph.add_memory(
                    memory_instance=rf_1B_I_Y,
                    operands=("I2",),
                    port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
                    served_dimensions=set(),
                )
                # Register file for input
                memory_hierarchy_graph.add_memory(
                    memory_instance=rf_1B_I_X,
                    operands=("I1",),
                    port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
                    served_dimensions=set(),
                )
            else:
                # Register file for weight
                memory_hierarchy_graph.add_memory(
                    memory_instance=rf_1B_W,
                    operands=("I2",),
                    port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
                    served_dimensions=set(),
                )
                # Register file for input
                memory_hierarchy_graph.add_memory(
                    memory_instance=rf_1B_I,
                    operands=("I1",),
                    port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
                    served_dimensions=set(),
                )
                # First SRAM for weights
                memory_hierarchy_graph.add_memory(
                    memory_instance=l1_w,
                    operands=("I2",),
                    port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
                    served_dimensions="all",
                )
            # Register file for output
            memory_hierarchy_graph.add_memory(
                memory_instance=rf_4B,
                operands=("O",),
                port_alloc=(
                    {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},
                ),
                served_dimensions=set(),
            )
            operands_act_mem=("I1","O","I2") if no_of_inputs>1 else ("I1","O")
            port_alloc_act_mem=[
                {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None},
                {
                    "fh": "rw_port_1",
                    "tl": "rw_port_1",
                    "fl": "rw_port_2",
                    "th": "rw_port_2",
                },
            ]
            if no_of_inputs>1:
                port_alloc_act_mem.append({"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None})
            # First SRAM for inputs and outputs
            memory_hierarchy_graph.add_memory(
                memory_instance=shared_l1,
                operands=operands_act_mem,
                port_alloc=tuple(port_alloc_act_mem),
                served_dimensions="all",
            )

            memory_hierarchy_graph.add_memory(
                memory_instance=l2,
                operands=("I1", "I2","O"),
                port_alloc=(
                    {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
                    {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
                    {
                        "fh": "w_port_1",
                        "tl": "r_port_1",
                        "fl": "w_port_1",
                        "th": "r_port_1",
                    },
                ),
                served_dimensions="all",
            )

            return memory_hierarchy_graph


        def get_operational_array():
            """Multiplier array variables"""
            multiplier_input_precision = [8, 8]
            multiplier_energy = 0.04
            multiplier_area = 1
            dimensions = {"D1": 16, "D2": 16}  # {'D1': ('OX', 16), 'D2': ('K', 16)}
            multiplier = Multiplier(
                multiplier_input_precision, multiplier_energy, multiplier_area
            )
            multiplier_array = MultiplierArray(multiplier, dimensions)

            return multiplier_array


        def get_dataflows():
            return [{"D1": ("OX", 16), "D2": ("K", 16)}]


        def get_core(id,operands):
            operational_array = get_operational_array()
            #get the memory hierarchy, from the l2 to the register level
            memory_hierarchy = get_memory_hierarchy(operational_array,len(operands))
            dataflows = get_dataflows()
            core = Core(id, operational_array, memory_hierarchy, dataflows)
            return core
        
        zigzag_operands = ["I","O","W"] if any([op in match_node.ops_occurrences for op in ["conv2d","dense"]]) else ["X","Y","O"]
        cores = {get_core(1,zigzag_operands)}
        global_buffer = None
        acc_name = 'MATCH'
        return Accelerator(acc_name, cores)