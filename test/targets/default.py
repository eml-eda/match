from match.target.target import MatchTarget


class DefaultExample(MatchTarget):
    def __init__(self):
        super(DefaultExample,self).__init__([],name="basic_cpu")
        # self.cpu_type = "arm_cpu"
        self.timestamp_to_ms = ""
        self.static_mem_plan = False
        self.static_mem_plan_algorithm = "hill_climb"