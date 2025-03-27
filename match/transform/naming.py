import tvm
from tvm import relay

# took from F.Daghero
@tvm.transform.module_pass(opt_level=0, name="MatchRenameIO")
class MatchRenameIO:
    def to_c_variable_name(self, name):
        """Convert Relay variable name to C-compliant name."""
        # Remove invalid characters
        import re
        name = re.sub(r'[^0-9a-zA-Z_]', '_', name)
        if len(name)==0:
            name = "match_inp"
        if name.isdigit() or name[0].isdigit():
            name = "match_inp_"+str(name)
        return name

    def transform_module(self, mod, ctx):
        """Module pass to rename input and output buffers to be C-compliant."""
        func = mod["main"]
        new_params = []
        name_map = {}

        # Rename inputs
        for param in func.params:
            new_name = self.to_c_variable_name(param.name_hint)
            new_var = relay.var(new_name, type_annotation=param.type_annotation)
            new_params.append(new_var)
            name_map[param] = new_var

        # Rewrite function body with renamed variables
        new_body = relay.expr.bind(func.body, name_map)

        # Create a new function with renamed parameters
        new_func = relay.Function(new_params, new_body)
        
        return tvm.IRModule.from_expr(new_func)