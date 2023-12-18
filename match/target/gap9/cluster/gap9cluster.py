from typing import Dict
from match.target.gap9.cluster.network_transformations import network_transformations as gap9network_transformations
from match.target.gap9.cluster.partitioning_patterns import partitioning_patterns as gap9partitioning_patterns
from match.target.exec_module import ExecModule
from match.target.gap9.cluster.cost_model import cost_model

class Gap9Cluster(ExecModule):
    def __init__(self):
        super(Gap9Cluster, self).__init__(name="cluster")
    
    def optimal_spatial_mapping_def(self, pattern_name: str = "conv2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
        if pattern_name=='conv2d' and (dim_sizes['FY']*dim_sizes['FX'])==1:
            self.optimal_spatial_mapping = [
                ("OY",4),("OX",4),("K",4)
            ]
        elif pattern_name=='conv2d' and layer_attrs["nn.conv2d_depthwise"]:
            self.optimal_spatial_mapping = [
                ("K",8),("OX",4),("OY",self.FULL_DIM)
            ]
        elif pattern_name=="conv2d":
            self.optimal_spatial_mapping = [
                ("OY",8),("OX",2),("K",4)
            ]
        elif pattern_name=='add':
            self.optimal_spatial_mapping = [
                ("OY",8),("OX",2)
            ]
        elif pattern_name=='dense':
            # TODO: K 8 C 1
            self.optimal_spatial_mapping = [
                ("K",8),("C",2)
            ]
        else:
            # DEFAULT LIKE CONV2D
            self.optimal_spatial_mapping = [
                ("OY",8),("OX",2),("K",4)
            ]

    def partitioning_patterns(self):
        return gap9partitioning_patterns()

    def network_transformations(self,opts):
        return gap9network_transformations(opts=opts)
    
    def cost_model(self):
        return cost_model()

    def include_list(self):
        return ["cluster_mem.h","cluster_comp.h"]