from typing import List

class MatchOp:
    def __init__(self, out_arr: List=[], var_arr: List=[], const_arr: List=[], op: str="ReLU", **kwargs) -> None:
        self.outs = out_arr
        self.vars = var_arr
        self.consts = const_arr
        self.op_code = 0
        self.op = op
        self.__dict__.update(kwargs)

    @property
    def attrs(self):
        return {k:self.__dict__[k] for k in self.__dict__.keys() - {"outs", "vars", "consts", "op_code", "op"}}
    
    @property
    def c_attrs(self):
        def _convert_to_c_array(value):
            if isinstance(value, (list, tuple)):
                return f"{{{', '.join(map(str, value))}}}"
            return value
        return {k: v if not isinstance(v, (list, tuple, bool)) else int(v) if isinstance(v,bool) else _convert_to_c_array(v)
                for k, v in self.attrs.items() if isinstance(v, (int, float, str, bool, list, tuple))}

        