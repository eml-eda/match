
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
    
    def partitioning_patterns(self):

        def dense_pt():
            """Create pattern for matmul."""
            dense = is_op("nn.dense")(
                wildcard(), is_constant()
            )
            return dense

        def only_int32_dense(node):
            # node is already just the dense TVM Call Node
            dense = node
            if dense.checked_type.dtype != "int32":
                return False
            return True

        return [
            PartitioningPattern(
                name="DENSE_PT",
                pattern=dense_pt,
                additional_checks=only_int32_dense
            ),
        ]

    def comp_apis_def(self, computational_apis: ComputationalApis=None, pattern_name = "dense"):
        computational_apis.compute_tile = "carus_compute_wrapper"
        return computational_apis
    
    def update_constants(self, match_node: MatchNode=None, pattern_name: str="dense"):
        for w_tensor in match_node.const_tensors.values():
            if "dense" in w_tensor.name:
                if w_tensor.layout!="CN":
                    w_tensor.data = w_tensor.data.transpose(1,0)
                    w_tensor.dims = [w_tensor.dims[1], w_tensor.dims[0]]
                w_tensor.layout = "CN"
                
    def include_list(self):
        return ["carus_helper/carus_helper"]