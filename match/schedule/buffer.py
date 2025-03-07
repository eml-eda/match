

class MatchMemBuffer:
    def __init__(self, name: str="im2col", mem_name: str="L1_CACHE", num_bytes: int=8):
        self.name = name
        self.mem_name = mem_name
        self.num_bytes = num_bytes
