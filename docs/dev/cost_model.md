# Cost Model

Currently MATCH supports only ZigZag as the scheduling engine. MATCH provides the user a predefined cost model that guides the engine research, however the user can amplify its capabilities by extending our simple implementation defined in `ZigZagMatchCostModel`.

This class gathers the default memory transfer costs for the inputs(and weights) in the variable `self.input_transfer_costs`, which is a dictionary that for each operand contains the list of transfer cost for that memory level.

The output costs are instead stored into the variable `self.output_transfer_costs`, that holds the list of costs.

This class saves also sizes of the loops into `self.loop_sizes`, and also the associated LayerData structure in `self.layer_data`. The user can exploit all this information to compute the overall costs. The user is advised to extend `def_transfer_cost` and `def_innermost_loops_cost`, where the first one should return a dictionary containing all the operands and the cost for transfering an inner tile. The latter instead refers to cost of the computation of a tile.

Importantly, `def_transfer_cost` is always called prior to `def_innermost_loops_cost`, so the user can set some parameters in the former, and use them later for the loops cost calculation.

For example, the following code defines the cost model of an accelerator, for which each transfer has an overhead of 100 cycles w.r.t. the cycles lost on the transfer itself, and where the number of cycles lost on the innemorst computations equals to 10000:
```python
from match.cost_model.zigzag import ZigZagMatchCostModel

class ExampleCostModel(ZigZagMatchCostModel):
    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
    ):
        super(ExampleCostModel,self).__init__(
            accelerator=accelerator,layer=layer,spatial_mapping=spatial_mapping,
            temporal_mapping=temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access,
            has_any_additional_buffer=True
        )
    
    def def_transfer_cost(self):
        return {operand:100 for operand in self.operands}
    
    def def_innermost_loops_cost(self):
        return 10000

class ExampleDigitalModule(ExecModule):
    ...
    def zigzag_cost_model(self):
        return ExampleCostModel
```