from match.tensor.tensor import MatchTensor

class MatchMemTransfer:
    def __init__(self, tensor: MatchTensor=MatchTensor(), top_mem: str="L2_MEM", mem: str="L1_SCRATCHPAD", sw_controlled: bool=False) -> None:
        self.tensor = tensor
        self.top_mem = top_mem
        self.mem = mem
        self.sw_controlled = sw_controlled