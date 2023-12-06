TWO_INPUTS_WORKLOADS=(
    "element_wise_sum"
)

class LayerData:
    def __init__(self):
        self.workload_name=""
        self.layer_arguments=[]
        self.operands=[]
        self.input_operands=[]
        self.padded_dims=[]
        self.input_dim_mapping=[]
        self.ordered_relevant_loops=dict()
        self.layer_attrs=dict()
        self.workload_name=""