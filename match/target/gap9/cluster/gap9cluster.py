from typing import Dict
from match.target.gap9.cluster.cost_model import Gap9ClusterCostModel
from match.target.gap9.cluster.network_transformations import network_transformations as gap9network_transformations
from match.target.gap9.cluster.partitioning_patterns import partitioning_patterns as gap9partitioning_patterns
from match.target.exec_module import ExecModule

class Gap9Cluster(ExecModule):
    def __init__(self):
        super(Gap9Cluster, self).__init__(name="cluster")
    
    def optimal_spatial_mapping_def(self, pattern_name: str = "conv2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
        if pattern_name=='gap9cluster_conv2d' and (dim_sizes['FY']*dim_sizes['FX'])==1:
            self.optimal_spatial_mapping = [
                ("OY",4),("OX",4),("K",4)
            ]
        elif pattern_name=='gap9cluster_conv2d' and layer_attrs["nn.conv2d_depthwise"]:
            self.optimal_spatial_mapping = [
                ("K",8),("OX",4),("OY",self.FULL_DIM)
            ]
        elif pattern_name=="gap9cluster_conv2d":
            self.optimal_spatial_mapping = [
                ("OY",8),("OX",2),("K",4)
            ]
        elif pattern_name=='gap9cluster_add':
            self.optimal_spatial_mapping = [
                ("OY",8),("OX",2)
            ]
        elif pattern_name=='gap9cluster_dense':
            # TODO: K 8 C 1
            self.optimal_spatial_mapping = [
                ("K",8),("C",2)
            ]
        else:
            # DEFAULT LIKE CONV2D
            self.optimal_spatial_mapping = [
                ("OY",8),("OX",2),("K",4)
            ]

    def memories_def(self, operands):
        super().memories_def(operands)
        self.platform_memories[0].double_buffering_support=True

        def buffers_for_l1_mem(layer_data,pattern_name):
            buff_mem=0
            # buffer for the cores of the accelerator (weights dimensions)
            if pattern_name!='gap9cluster_add' :
                buff_mem=2*layer_data.loop_dim_size['C']*layer_data.loop_dim_size['FY']*layer_data.loop_dim_size['FX']
            # buff for each core
            buff_mem*=8
            # bias
            if pattern_name!="gap9cluster_add":
                buff_mem+=layer_data.loop_dim_size['K']*4
            return buff_mem
        
        self.platform_memories[0].buffer_for_layer_func=buffers_for_l1_mem

    def partitioning_patterns(self):
        return gap9partitioning_patterns()

    def network_transformations(self,opts):
        return gap9network_transformations(opts=opts)

    def def_include_list(self):
        self.include_list+=["cluster_mem.h","cluster_comp.h"]

    def mem_apis_def(self):
        self.mem_apis.copy_out_curr_computation="cluster_copy_out_curr_computation"
        self.mem_apis.copy_out_prev_computation="cluster_copy_out_prev_computation"
        self.mem_apis.mem_transfer={
            "I":"cluster_mem_transfer_I",
            "X":"cluster_mem_transfer_X",
            "Y":"cluster_mem_transfer_Y",
            "W":"cluster_mem_transfer_W",
            "O":"cluster_mem_transfer_O",
        }
        self.mem_apis.pattern_constants_loading="cluster_pattern_constant_loading"
        self.mem_apis.pointer_offset={
            "O":"cluster_pointer_offset_O",
            "I":"cluster_pointer_offset_I",
            "X":"cluster_pointer_offset_X",
            "Y":"cluster_pointer_offset_Y",
            "W":"cluster_pointer_offset_W",
        }
        self.mem_apis.shutdown_mem="cluster_shutdown_mem"
        self.mem_apis.startup_memory_and_set_pattern="cluster_startup_memory_and_set_pattern"

    def comp_apis_def(self):
        self.comp_apis.innermost_computation="cluster_kernel_function_wrapper"
        self.comp_apis.init_other_kernel_params="cluster_init_other_kernel_params"
    
    def platform_apis_def(self):
        self.platform_apis.init_platform="cluster_init_platform"
    
    def sync_apis_def(self):
        self.sync_apis.async_transfers="cluster_wait_any_transfer"
        self.sync_apis.curr_computation="cluster_wait_curr_computation"
        self.sync_apis.prev_computation="cluster_wait_prev_computation"

    def types_def(self):
        self.types.kernel_struct="cluster_kernel"
        self.types.mem_data_macro_and_type="GAP_L2_DATA uint8_t"

    def cost_model(self):
        return Gap9ClusterCostModel