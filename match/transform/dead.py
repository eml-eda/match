from tvm import relay

def is_identity_function(func, mod):
    # Check that the function has one parameter and its body is just that parameter.
    if hasattr(func, "op") and isinstance(func.op, relay.GlobalVar):
        func = mod[func.op.name_hint]
        if len(func.params) == 1:
            param = func.params[0]
            # In practice, you might also check that this function is a BYOC function
            # (e.g., by examining its attributes such as Compiler="match")
            return isinstance(func.body, relay.Var) and func.body.name_hint == param.name_hint
    return False

@relay.transform.function_pass(opt_level=1)
class MatchRemoveIdentityBYOC(relay.ExprMutator):

    def __init__(self):
        super().__init__()

    def transform_function(self, func, mod, ctx):
        self.mod = mod
        func = self.visit(func)
        return func
    
    def visit_call(self, call):
        new_call = super().visit_call(call)
        # If the call is to a function, check if it is an identity function.
        if is_identity_function(new_call, self.mod):
            # Replace the call with the argument directly.
            return new_call.args[0]
        return new_call