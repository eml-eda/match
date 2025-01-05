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
        def _convert_to_c(value):
            if isinstance(value,str):
                return '"'+value+'"'
            if isinstance(value,bool):
                return int(value)
            if isinstance(value,(int,float)):
                return value
            if isinstance(value,(list,tuple)):
                return _convert_to_c_array(value)
            return value
        return {k: _convert_to_c(v) for k, v in self.attrs.items() if isinstance(v, (int, float, str, bool, list, tuple))}
    
    def basic_schedules(self):
        return []

        