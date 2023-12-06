from typing import Dict
from match.hwmodel.gap9cluster.network_transformations import network_transformations as gap9network_transformations
from match.hwmodel.gap9cluster.partitioning_patterns import partitioning_patterns as gap9partitioning_patterns
from match.hwmodel.hwmodel import HwModel
from match.hwmodel.gap9cluster.cost_model import cost_model

class Gap9ClusterHwModel(HwModel):
    def __init__(self):
        super(Gap9ClusterHwModel, self).__init__()
    
    def optimal_spatial_mapping_def(self, workload_name: str = "conv2d",dim_sizes:Dict[str,int]={}):
        if workload_name=='conv_2d' and (dim_sizes['FY']*dim_sizes['FX'])==1:
            self.optimal_spatial_mapping = [
                ("OY",4),("OX",4),("K",4)
            ]
        elif workload_name=='conv_2d':
            self.optimal_spatial_mapping = [
                ("OY",8),("OX",2),("K",4)
            ]
        elif workload_name=="depthwise_conv_2d":
            self.optimal_spatial_mapping = [
                ("K",8),("OX",4),("OY",self.FULL_DIM)
            ]
        elif workload_name=='element_wise_sum':
            self.optimal_spatial_mapping = [
                ("OY",8),("OX",2)
            ]
        elif workload_name=='dense':
            # TODO: K 8 C 1
            self.optimal_spatial_mapping = [
                ("K",8),("C",2)
            ]
    
    #def memories_def(self, operands):
    #    return super().memories_def(operands)

    def partitioning_patterns(self):
        return gap9partitioning_patterns()

    def network_transformations(self,opts):
        return gap9network_transformations(opts=opts)
    
    def cost_model(self):
        return cost_model()