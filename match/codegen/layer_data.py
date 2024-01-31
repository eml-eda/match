TWO_INPUTS_WORKLOADS=(
    "element_wise_sum"
)

class LayerData:
    def __init__(self):
        self.layer_arguments=[]
        self.operands=[]
        self.input_operands=[]
        self.padded_dims=[]
        self.input_dim_mapping=[]
        self.ordered_relevant_loops=dict()
        self.layer_attrs=dict()
        self.pattern_operations=[]
        self.specific_pattern=""
        self.equation="O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]"
        self.strides = [1, 1]
        self.dilations = [1, 1]
        self.visited_operations=list()
        self.loop_dim_size = {
            "B": 1,
            "K": 1,
            "C": 1,
            "OY": 1,
            "OX": 1,
            "FY": 1,
            "FX": 1,
        }
        self.dimension_relations = [
            "ix=1*ox+1*fx",
            "iy=1*oy+1*fy",
        ]
        self.operand_precision = {
            "O": 8,
            "O_final": 8,
            "W": 8,
            "I": 8,
        }
        self.padding = {
            "IY": (0, 0),
            "IX": (0, 0),
        }
        self.pr_loop_dim_size = {"IY": 1, "IX": 1}
        self.operand_source_dimension_mapping={"I": {"IX": "OX", "IY": "OY"}}
        self.constant_operands=["W"]
        self.operand_source={"W": [], "I": []}

    def __eq__(self,other):
        return self.operands==other.operands and self.input_operands==other.input_operands \
        and self.specific_pattern==other.specific_pattern \
        and self.padded_dims==other.padded_dims and self.input_dim_mapping==other.input_dim_mapping \
        and self.ordered_relevant_loops==other.ordered_relevant_loops and self.pattern_operations==other.pattern_operations \
        and self.equation==other.equation and self.strides==other.strides and self.dilations==other.dilations \
        and self.loop_dim_size==other.loop_dim_size and self.dimension_relations==other.dimension_relations \
        and self.operand_precision==other.operand_precision and self.padding==other.padding \
        and self.pr_loop_dim_size==other.pr_loop_dim_size and self.operand_source_dimension_mapping==other.operand_source_dimension_mapping \
        and self.constant_operands==other.constant_operands and self.operand_source==other.operand_source \
        and self.layer_attrs==other.layer_attrs
    