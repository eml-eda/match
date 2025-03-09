
from abc import ABC,abstractmethod
from math import ceil
from typing import Any, Dict,List,Type
from match.cost_model.zigzag import ZigZagMatchCostModel, ZigZagMatchNoTilingCostModel
from match.partition.partitioning_pattern import PartitioningPattern
from match.target.memory_inst import MemoryInst
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant
import tvm
import numpy as np
import sys

class MemoryApis:
    """All the memory APIs that are used by MATCH on the templates
    """
    def __init__(self):
        # these APIS are used for peculiar layouts of fused tensors
        # which are not currently supported, for instance if a tensor
        # needs to be fused and they have to be contiguous or they
        # split by channels or other stuff these APIS must be implemented
        # to compute the size of the tensor(ex. redundant data so MATCH cannot get accurate size)
        # and the pt, since the dimensions are all interleaved
        self.get_size_of_fused_tensor = "" # TODO: test
        self.get_pt_of_fused_tensor = "" # TODO: test
        # API to handle the SW-controlled mem transfers
        # this API receives all the info about the tensor,
        # the type of op(LOAD/STORE), the top and lower memories names
        # and the overall context
        self.mem_transfer = ""
        # these dict respectively for each SW controlled memory store
        # allocate the memory first and use it throughout the node
        # finally deallocating at the end, the memory pt is kept by MATCH
        self.free_memory = dict()
        self.init_memory = dict()
        # API used to signal to the exec module about a required buffer
        # MATCH decides the pointer but it doesn't store it
        # the exec module lib should keep it
        self.alloc_buffer = ""
        """
        APIs and flags from the legacy lib
        DEFAULT_LAYOUT = {"O":"NCHW","I":"NCHW","W":"NCHW","X":"NCHW","Y":"NCHW"}
        self.startup_memory="match_startup_memory"
        self.shutdown_mem="match_shutdown_mem"
        self.copy_out_curr_computation="match_copy_out_curr_computation"
        self.copy_out_prev_computation="match_copy_out_prev_computation"
        self.pointer_offset={
            "O":f"match_pointer_offset_{layout_per_operand['O'] if 'O' in layout_per_operand else DEFAULT_LAYOUT['O']}_O",
            "W":f"match_pointer_offset_{layout_per_operand['W'] if 'W' in layout_per_operand else DEFAULT_LAYOUT['W']}_W",
            "I":f"match_pointer_offset_{layout_per_operand['I'] if 'I' in layout_per_operand else DEFAULT_LAYOUT['I']}_I",
            "X":f"match_pointer_offset_{layout_per_operand['X'] if 'X' in layout_per_operand else DEFAULT_LAYOUT['X']}_X",
            "Y":f"match_pointer_offset_{layout_per_operand['Y'] if 'Y' in layout_per_operand else DEFAULT_LAYOUT['Y']}_Y",
        }
        self.mem_transfer_old={
            "O":"match_mem_transfer_O",
            "W":"match_mem_transfer_W",
            "I":"match_mem_transfer_I",
            "X":"match_mem_transfer_X",
            "Y":"match_mem_transfer_Y",
        }
        self.pattern_constants_loading="match_pattern_constants_loading"
        """

class ComputationalApis:
    """All the APIs relating to the computational part that are used later by MATCH templates
    """
    def __init__(self):
        # API to handle computation to the specific back-end
        self.compute_tile = ""
        """
        APIs and flags from the legacy lib
        self.init_other_kernel_params="match_init_other_kernel_params"
        self.innermost_computation="match_innermost_computation"
        self.specific_pattern=pattern_name
        """

class PlatformApis:
    """All the APIs for the management of the platform that are used by templates of MATCH
    """
    def __init__(self):
        # API used to pass from the host to the exec module controller
        # if needed, certain programming modes require this
        self.init_platform=""
        # API used to differentiate between each parallel
        # head of the exec module, used if the parallel mode
        # is enabled with parallelize task
        self.get_task_id = "" # TODO: test
        self.parallelize_task = "" # TODO: test
        # signals to the exec module that it is used
        # and then freed
        self.init_module = ""
        self.free_module = ""

        """
        APIs and flags from the legacy lib
        self.init_platform_need_kernel_data=False
        self.set_task_id="match_task_id"
        """

class SyncApis:
    """All the APIs for the synchronization that are used by templates of MATCH
    """
    def __init__(self):
        # flags to signal type of memory system --> async or not
        self.must_sync_after_load = False
        self.must_sync_after_store = False
        self.must_sync_after_computation = False
        # more complex sync since computation is parallelized
        self.wait_parallel_tasks = "" # TODO: test
        # synchronization after the computation of the tile
        self.wait_tile_computation = ""
        # basic wait for load and store memory movements
        # they can be the same API
        self.wait_load = ""
        self.wait_store = ""
        # more complex sync since computation is buffered with
        # memory movement
        self.wait_buffer_parallel_tasks = "" # TODO: test
        self.wait_buffer_tile_computation = "" # TODO: test
        """
        unused APIs
        self.wait_prev_tile_computation = "" # not used
        self.wait_parallel = "" # not used
        self.wait_store_prev_tile = "" # not used
        """
        """
        APIs from the legacy lib
        self.async_transfers="match_async_transfers"
        self.prev_computation="match_prev_computation"
        self.curr_computation="match_curr_computation"
        self.sync_multilevel_transfer="match_sync_multilevel_transfer"
        self.wait_input_transfers="match_wait_input_transfers"
        self.wait_output_transfers="match_wait_output_transfers"
        """

# unused currently
class MatchTypes:
    """MACROS and types that can be used by MATCH
    """
    def __init__(self):
        self.mem_data_macro_and_type="unsigned int"
        self.kernel_struct="match_kernel"

class ExecModule(ABC):
    """Unit that will handle the compuation of a layer
    """
    def __init__(self,name:str="default_exec_module",
                 specific_patterns=[],
                 src_path:str="",
                 inc_path:str="",
                 **kwargs):
        self.name=name
        self.FULL_DIM = sys.maxsize
        self.zigzag_optimal_spatial_mapping = None
        self.platform_memories = None
        self.specific_patterns=specific_patterns
        self.specific_pattern=""
        self.src_path=src_path
        self.inc_path=inc_path
        self.module_options=dict()
        self.backend = "ZigZag"
        # currently only ZigZag has been actually tested
        self.schedule_engine = "ZigZag"

    def backend_constraints_check(self,match_node,schedule,block,lp,lp_idx):
        # if any([mt.sw_controlled for mt in lp.mem_transfers]):
            # return False
        # else:
        for lp_after in block.loops[lp_idx+1:]:
            if any([mt.sw_controlled for mt in lp_after.mem_transfers]):
                return False
        return True


    def partitioning_patterns(self):
        return []
    
    def network_transformations(self,opts):
        return []

    def memories_def(self,pattern_name,operands):
        """define the memory hierarchy of the unit by setting self.platform_memories

        Args:
            operands (List[Str]): list of operands
        """
        return [
            # from lower level to higher level memories
            # TEST: set now L1 to 9 kB just to force TILING 
            MemoryInst(name="l1_mem",k_bytes=90,operands=operands,double_buffering_support=True),
            MemoryInst(name="l2_mem",k_bytes=1408,operands=operands,r_ports=1,w_ports=1,rw_ports=0),
        ]

    @property
    def mem_hierarchy(self):
        memories_ = self.get_all_memories()
        return {
            "out":memories_,
            "var":memories_,
            "const":memories_,
            "inter":memories_,
        }

    def get_all_memories_names(self):
        return [m.name for m in self.memories_def(pattern_name="conv2d",operands=["O","I","W"])]
    
    def get_all_memories(self):
        return [m for m in self.memories_def(pattern_name="conv2d",operands=["O","I","W"])]
    
    @property
    def memories(self):
        return self.get_all_memories_names()
    
    def match_memories(self,pattern_name,operands):
        self.platform_memories=self.memories_def(pattern_name,operands)

    def limit_spatial_mapping_to(self,dim_size:int=1,optimal_spat:int=1):
        # find greater common denominator that is smaller than the optimal spatial mapping
        spatial_dim=dim_size
        if dim_size>optimal_spat:
            for div_val in [_+1 for _ in range(optimal_spat)][::-1]:
                if (dim_size%div_val)==0:
                    spatial_dim=div_val
                    break
        return spatial_dim

    def zigzag_optimal_spatial_mapping_def(self, match_node = None, pattern_name: str = "conv_2d"):
        """Define the optimal spatial mapping for the current node

        Args:
            pattern_name (str, optional): analyzed pattern. Defaults to "conv_2d".
            dim_sizes (Dict[str,int], optional): sizes of each dimension. Defaults to {}.
            layer_attrs (Dict, optional): attributes specific to the layer. Defaults to {}.
        """
        return [ ("K",1), ("OY",1) ]

    def zigzag_set_optimal_spatial_mapping(self, match_node = None, pattern_name: str = "conv_2d",):
        self.zigzag_optimal_spatial_mapping=self.zigzag_optimal_spatial_mapping_def(match_node=match_node,pattern_name=pattern_name)
        if self.zigzag_optimal_spatial_mapping is None:
            self.zigzag_optimal_spatial_mapping = [ ("K",1), ]

    def specific_pattern_def(self, match_node = None, pattern_name: str = "conv_2d"):
        return pattern_name

    def match_specific_pattern(self, match_node = None, pattern_name: str = "conv_2d"):
        self.specific_pattern=self.specific_pattern_def(match_node=match_node,pattern_name=pattern_name)
        
    def get_optimal_spat_size(self,optimal_spat:int=1,dim_size:int=1):
        if optimal_spat==self.FULL_DIM:
            return dim_size
        else:
            return optimal_spat   
    
    # NOTE: not used, remove
    def adjust_dimensions_and_precision(self,loop_dim_size:Dict[str,int]={},pr_loop_dim_size:Dict[str,int]={},
                                        operand_precision:Dict[str,int]={}, strides:List[int]=[1,1], pattern_name:str="conv2d"):
        return loop_dim_size,pr_loop_dim_size,operand_precision,operand_precision
    
    def zigzag_cost_model(self):
        """Function that defines the used cost model to guide the schedule search

        Returns:
            Class: class itself(not an instance) of the used cost model
        """
        return ZigZagMatchCostModel if len(self.memories)>1 else ZigZagMatchNoTilingCostModel
    
    def constrain_schedule(self,schedule,match_node):
        return schedule
    
    def set_buffers_for_schedule(self, match_node, schedule, pattern_name, engine):
        return

    # NOTE: not used, remove
    def weights_and_constants(self,match_node,pattern_name):
        """define how the weights and constants of a layer must be saved in C on the generated code

        Args:
            layer_arguments (List, optional): Dict of the arguments(parameters) for the node. Defaults to [].
        """
        def c_friendly_npvalue(arr):
            # params: arr is expected to be a numpy version of the value, it should be an array but it may be also just a single value
            if len(arr.shape)>0:
                # this is actually an array and not a single value
                arr=arr.reshape([arr.shape[0]]).astype(np.uint8)
                return f'{{{str(list(arr))[1:len(str(list(arr)))-1]}}}'
            else:
                return str(arr)
        def bytaze(value):
            return np.frombuffer(value.tobytes(),dtype='uint8')
        arguments=np.array([],dtype=np.uint8)
        single_constants=dict()
        for (layer_arg_name,layer_arg_val) in match_node.const_tensors.items():
            if isinstance(layer_arg_val, tvm.relay.Constant):
                if len(layer_arg_val.data.shape)==0:
                    single_constants[layer_arg_name]=str(layer_arg_val.data)
                else:
                    constbytes=bytaze(layer_arg_val.data.numpy())
                    arguments=np.concatenate((arguments,constbytes))
        return {
            "value":c_friendly_npvalue(arguments),
            "len":arguments.shape[0],
            "shape":f"[{ceil(arguments.shape[0])}]",
            "single_costants":single_constants,
        }
    
    def update_constants(self, match_node, pattern_name):
        pass

    def additional_kernel_parameters(self,pattern_name):
        return dict()
    
    def layout_per_operand_def(self,pattern_name,specific_pattern,operands):
        return dict()

    def types_def(self,match_types: MatchTypes=MatchTypes()):
        return match_types

    def match_types(self, pattern_name):
        return self.types_def(MatchTypes(pattern_name=pattern_name))

    def mem_apis_def(self,memory_apis: MemoryApis=MemoryApis(), pattern_name: str="conv2d"):
        """Functions that set the memory related APIs of the unit
        """
        return memory_apis

    # all the APIS methods receive the pattern name since there could be
    # a difference between them(for depthwise conv2d data must be stored in a certain way etc.)
    def match_mem_apis(self,pattern_name):
        return self.mem_apis_def(MemoryApis(),pattern_name)
    
    def sync_apis_def(self,sync_apis: SyncApis=SyncApis(), pattern_name: str="conv2d"):
        """Functions that set the synchronization related APIs of the unit
        """
        return sync_apis
    
    def match_sync_apis(self,pattern_name):
        return self.sync_apis_def(SyncApis(), pattern_name)
    
    def comp_apis_def(self,computational_apis: ComputationalApis=ComputationalApis(), pattern_name: str="conv2d"):
        """Functions that set the computation related APIs of the unit
        """
        return computational_apis
    
    def match_comp_apis(self,pattern_name):
        return self.comp_apis_def(ComputationalApis(), pattern_name)
    
    def platform_apis_def(self,platform_apis: PlatformApis=PlatformApis(), pattern_name: str="conv2d"):
        """Functions that set the platform related APIs of the unit
        """
        return platform_apis
    
    def match_platform_apis(self,pattern_name):
        return self.platform_apis_def(PlatformApis(), pattern_name)

    def def_include_list(self):
        """Functions that sets the list of headers to include additionally in the template
        """
        return []
    
    def match_include_list(self,pattern_name):
        ex_module_inc_list=self.def_include_list(pattern_name)
        return ex_module_inc_list

    # NOTE: not used, remove
    def operand_memories(self,operands):
        return {
            operand:[mem.name for mem in self.platform_memories if operand in mem.operands][::-1]
            for operand in operands
        }
    
    def add_option_to_module(self,option,value):
        self.module_options[option]=value

    def adjust_network(self,opts):
        return []

    def zigzag_architecture(self, optimal_spatial_mapping = None, platform_memories = None, match_node = None):
        return None