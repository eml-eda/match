import copy
import functools
from math import ceil
import operator
import sys
from typing import Dict, List

import numpy as np
from match.target.diana.digital.cost_model import DigitalAcceleratorCostModel
from match.target.diana.digital.network_transformations import network_transformations as diana_digital_net_trans
from match.target.diana.digital.network_transformations import adjust_network as diana_digital_adj_net
from match.target.diana.digital.partitioning_patterns import partitioning_patterns as diana_digital_patterns
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

def depthwise_conv_2d_bias(value,loop_sizes):
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
    for ch in range(int((loop_sizes['K'] * 4 +15)/16)):
        for byte in range(4):
            final_weights.append(new_weights[(loop_sizes['K'] * 4 * byte + ch*16):(loop_sizes['K'] * 4 * byte + ch*16 + 16)])
    channels=ceil(np.asarray(final_weights).flatten().shape[0]/(16*4))
    return np.asarray(final_weights).flatten().reshape(channels,ceil(np.asarray(final_weights).flatten().shape[0]/channels)).tolist()

def dense_bias(value,loop_sizes):
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

def conv_2d_bias(value,loop_sizes):
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
    channels=loop_sizes['K']
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

def depthwise_conv_2d_weights(value,loop_sizes):
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
    channels=ceil(flatvalue.shape[0]/(16*loop_sizes['FY']*loop_sizes['FX']))
    return tempf.flatten().reshape(channels,ceil(flatvalue.shape[0]/channels)).tolist()

def conv_2d_weights(value,loop_sizes):
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

def dense_weights(value,loop_sizes):
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

def transform(key,value,loop_sizes,layer_name):
    if ''.join([n for n in layer_name.split('_') if n not in LAYERS_WITH_SAME_NAME]) in key:
        return (WEIGHT_ROUTER[layer_name if layer_name in WEIGHT_ROUTER else 'default'](value.data.numpy(),loop_sizes),key)
    elif 'bias' in key:
        return (BIAS_ROUTER[layer_name if layer_name in BIAS_ROUTER else 'default'](value.data.numpy(),loop_sizes),key)
    else:
        return (None,(key,c_friendly_npvalue(value.data.numpy())))

def transform_key_name(key: str):
    if 'shift' in key:
        return 'output_shift'
    else:
        return key



class DigitalAccelerator(ExecModule):
    def __init__(self):
        super(DigitalAccelerator, self).__init__(name="digital",
                                          specific_patterns=[
                                              "conv_2d",
                                              "dense",
                                              "depthwise_conv_2d",
                                              "element_wise_sum",
                                          ],
                                          src_path=os.path.dirname(__file__)+"/src",
                                          inc_path=os.path.dirname(__file__)+"/include")

    def optimal_spatial_mapping_def(self, pattern_name: str = "conv2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
        return [
                ("K",16),("OX",16),
            ]
    
    def specific_pattern_def(self, pattern_name: str = "conv_2d", dim_sizes: Dict[str, int] = ..., layer_attrs: Dict = ...):
        if pattern_name=="conv2d" and layer_attrs["nn.conv2d_depthwise"]:
            return "depthwise_conv_2d"
        else:
            return pattern_name

    def memories_def(self, pattern_name, operands):
        mem = [
            # from lower level to higher level memories
            MemoryInst(name="act_mem",k_bytes=256,operands=operands,double_buffering_support=False),
            MemoryInst(name="dram",k_bytes=512,operands=operands,r_ports=1,w_ports=1,rw_ports=0),
        ]
        if "W" in operands:
            mem = [
                MemoryInst(name="weight_mem",k_bytes=64,operands=["W"],double_buffering_support=False),
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

    def cost_model(self):
        return DigitalAcceleratorCostModel
    
    #def layout_per_operand_def(self, pattern_name, specific_pattern, operands):
    #    return {operand:"NHWC" for operand in operands}
    
    #def adjust_network(self, opts):
    #    return gap_adjust_net(opts=opts)
    
    def weights_and_constants(self,pattern_name,layer_data,layer_arguments:List=[]):
        """define how the weights and constants of a layer must be saved in C on the generated code

        Args:
            layer_arguments (List, optional): Dict of the arguments(parameters) for the node. Defaults to [].
        """
        output_channels=layer_data.loop_dim_size['K'] if 'depthwise' not in pattern_name else layer_data.loop_dim_size['K']*4
        constants=[transform(k,v,layer_data.loop_dim_size,pattern_name) for k,v in layer_arguments.items() if isinstance(v,tvm.relay.Constant)]
        diana_weights=[]
        for ch in range(ceil(output_channels/16)):
            for (c,keyval) in constants:
                if c is not None:
                    diana_weights+=c[ch]
                elif c is None and ch==0:
                    layer_data.layer_attrs[transform_key_name(keyval[0])]=keyval[1]
        diana_weights=(np.asarray(diana_weights).flatten())
        return {
            'value':c_friendly_npvalue(diana_weights),
            'type':'char',
            'shape':f'[{ceil(diana_weights.shape[0])}]'
        }