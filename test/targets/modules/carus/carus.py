
import os
from match.target.exec_module import ExecModule, ModuleLib
from match.partition.partitioning_pattern import PartitioningPattern
from match.target.exec_module import ComputationalApis
from match.node.node import MatchNode
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant

class Carus(ExecModule):
    def __init__(self):
        super().__init__(
            name = "carus",
            libs_required = {
                "carus_helper": ModuleLib(name="carus_helper", base_path=os.path.dirname(__file__)+"/../libs/carus_helper"),
            },
        )

    def module_memories(self):
        return []
    
    def partitioning_patterns(self):

        def dense_bnrom_fake_requant_pt():
            """Create pattern for conv2D with optional fused relu."""
            dense = is_op("nn.dense")(
                wildcard(), wildcard()
            )
            scale = is_op("multiply")(dense, wildcard()) | is_op("multiply")(wildcard(), dense)
            bias = is_op("add")(scale, is_constant()) | is_op("add")(is_constant(), scale)
            right_shift = is_op("right_shift")(bias, is_constant())
            clip = is_op("clip")(right_shift)
            return clip

        def dense_add_pt():
            """Create pattern for conv2D with optional fused relu."""
            dense = is_op("nn.dense")(
                wildcard(), wildcard()
            )
            bias = is_op("add")(dense, is_constant()) | is_op("add")(is_constant(), dense)
            return bias

        def additional_checks_fake_requant(node):
            return False
        
        def additional_checks_dense_add(node):
            return False

        return [
            PartitioningPattern(name="DENSE_BNORM_FAKE_REQUANT_PT", pattern=dense_bnrom_fake_requant_pt,
                                # additional_checks=additional_checks_fake_requant
                                ),
            PartitioningPattern(name="DENSE_ADD_PT", pattern=dense_add_pt,
                                # additional_checks=additional_checks_dense_add
                                ),
        ]

    def comp_apis_def(self, computational_apis: ComputationalApis=None, pattern_name = "conv2d"):
        computational_apis.compute_tile = "carus_compute_wrapper"
        return computational_apis
    
    def update_constants(self, match_node: MatchNode=None, pattern_name: str="conv2d"):
        for w_tensor in match_node.const_tensors.values():
            if "dense" in w_tensor.name:
                if w_tensor.layout!="CN":
                    w_tensor.data = w_tensor.data.transpose(1,0)
                    w_tensor.dims = [w_tensor.dims[1], w_tensor.dims[0]]
                w_tensor.layout = "CN"
                
    def include_list(self):
        return ["carus_helper/carus_helper"]