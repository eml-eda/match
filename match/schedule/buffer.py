

class MatchMemBuffer:
    def __init__(self, name: str="im2col", mem_name: str="L1_CACHE", num_bytes: int=8, required: bool=True) -> None:
        """
        :param name: Name of the buffer
        :param mem_name: Memory where the buffer is allocated
        :param num_bytes: Number of bytes allocated for the buffer
        :param required: If the buffer is required for the schedule
        """
        self.name = name
        self.mem_name = mem_name
        self.num_bytes = num_bytes
        self.required = required
