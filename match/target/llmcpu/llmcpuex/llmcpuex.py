import copy
import functools
from math import ceil
import operator
import sys
from typing import Any, Dict, List

import numpy as np
from match.target.diana.digital.cost_model import DigitalAcceleratorCostModel
from match.target.diana.digital.network_transformations import network_transformations as diana_digital_net_trans
from match.target.diana.digital.network_transformations import adjust_network as diana_digital_adj_net
from match.target.diana.digital.partitioning_patterns import partitioning_patterns as diana_digital_patterns
from match.target.exec_module import ExecModule, PlatformApis, MemoryApis, SyncApis, ComputationalApis, MatchTypes
import os
from match.target.memory_inst import MemoryInst
import tvm
from tvm.relay.dataflow_pattern import wildcard, is_op
from match.partition.partitioning_pattern import PartitioningPattern
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.dataflow_pattern import DFPatternCallback, rewrite, wildcard, is_op, is_constant

class LLMCpuEx(ExecModule):
    def __init__(self,**kwargs):
        super(LLMCpuEx, self).__init__(name="llmcpuex",
                                          specific_patterns=[
                                              "dense",
                                          ],
                                          src_path=os.path.dirname(__file__)+"/src",
                                          inc_path=os.path.dirname(__file__)+"/include",
                                          **kwargs)
        

    def optimal_spatial_mapping_def(self, pattern_name: str = "conv2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
        return [
                ("K",16)
            ]

    def memories_def(self, pattern_name, operands):
        mem = [
            # from lower level to higher level memories
            MemoryInst(name="L2_CACHE",k_bytes=32*1024,operands=operands,r_bw=64*8,w_bw=64*8,r_ports=0,w_ports=0,rw_ports=2,double_buffering_support=False),
            MemoryInst(name="SDRAM",k_bytes=8*1024*1024,operands=operands,r_ports=1,w_ports=1,rw_ports=0,r_bw=16*8,w_bw=16*8),
        ]
        return mem
    
    def partitioning_patterns(self):
        def dense_pattern():
            return is_op("nn.dense")(
                wildcard(), wildcard()
            )
        return [
            #PartitioningPattern(name="dense", pattern=dense_pattern, ordered_operation="dense"),
        ]

    def network_transformations(self, opts):
        class RemoveBoolCastPattern(DFPatternCallback):
            def __init__(self, require_type=False):
                super().__init__(require_type)
                self.pattern = is_op("cast")(wildcard()).has_attr({"dtype":"bool"})

            def callback(self, pre, post, node_map):
                return pre.args[0]

        @tvm.ir.transform.module_pass(opt_level=0)
        class RemoveBoolCastTransform:
            def transform_module(
                self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
            ) -> tvm.ir.IRModule:
                for global_var, func in mod.functions.items():
                    func = rewrite(RemoveBoolCastPattern(), func)
                    mod.update_func(global_var, func)
                return mod

            def __call__(self, mod):
                return self.transform_module(mod)
            
        return [RemoveBoolCastTransform()]

    def def_include_list(self,patter_name):
        return ["llmcpulib.h"]

    def comp_apis_def(self,comp_apis: ComputationalApis=ComputationalApis()):
        comp_apis.innermost_computation="llmkernel_wrapper"
        comp_apis.specific_pattern=self.specific_pattern
        return comp_apis
    
    #def cost_model(self):
    #    return LLMAcceleratorCostModel