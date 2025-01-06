import ast
import astor
import inspect
import astunparse


# class NeoptIRi32Value:
#     def __init__(self, value: int):
#         self.value = value

#     def __add__(self, other):
#         return NeoptIRi32Value(self.value + other.value)

#     def __mul__(self, other):
#         return NeoptIRi32Value(self.value * other.value)

#     def __getitem__(self, key):
#         return self.value
    
# class NeoptIRi8Tensor:
#     def __init__(self, shape: Tuple[int]):
#         self.shape = shape

#     def __getitem__(self, key):
#         return NeoptIRTensorGetAt(self, key)
    
# class NeoptIRi32Tensor:
#     def __init__(self, shape: Tuple[int]):
#         self.shape = shape

#     def __getitem__(self, key):
#         return NeoptIRTensorGetAt(self, key)

# class PulpTarget(NeoptTarget):    
#     def __init__(self):
#         super(PulpTarget, self).__init__(backend_ops = [self.pulp_vect_mac])

#     def backend_operations(self):
#         def pulp_vect_mac(out: NeoptIRi32Value, inp: NeoptIRi8Tensor, weights: NeoptIRi8Tensor):
#             """Create a PULP MAC operation.
#             """
#             for i in range(4):
#                 out += inp[i] * weights[i]
#             return out
#         self.backend_operations = 

# def fully_connected_i8_layer(out: NeoptIRi32Tensor, inp: NeoptIRi8Tensor, weights: NeoptIRi8Tensor):
#     """Create a fully connected layer with 4 input and output channels.
#     """
#     for i in range(weights.shape[0]):
#         for j in range(weights.shape[1]):
#             out[i] += inp[j] * weights[i][j]
#     return out

def fully_connected_i8_layer(out, inp, weights):
    """Create a fully connected layer with 4 input and output channels.
    """
    for i in range(weights.shape[0]):
        for j in range(4):
            out[i] += inp[j] * weights[i][j]
    return out

def match_and_rewrite(layer_func, sub_func):
    """
    Match and rewrite the layer function if a subpart matches the subfunction.
    :param layer_func: The main layer function to be matched and rewritten.
    :param sub_func: The subfunction to match within the layer function.
    :return: The rewritten layer function if a match is found, otherwise the original layer function.
    """

    class SubFuncMatcher(ast.NodeVisitor):
        def __init__(self, sub_func_ast):
            self.sub_func_ast = sub_func_ast
            self.matched = False

        def visit_FunctionDef(self, node):
            if not self.matched and ast.dump(node) == ast.dump(self.sub_func_ast):
                self.matched = True
                self.generic_visit(node)
            else:
                self.generic_visit(node)

    layer_func_ast = ast.parse(inspect.getsource(layer_func))
    sub_func_ast = ast.parse(inspect.getsource(sub_func)).body[0]
    sub_func_ast = ast.parse(astor.to_source(sub_func_ast)).body[0]

    # Print the ASTs for debugging
    print("Layer Function AST:")
    print(ast.dump(layer_func_ast, indent=4))
    print("\nSub Function AST:")
    print(ast.dump(sub_func_ast, indent=4))

    matcher = SubFuncMatcher(sub_func_ast)
    matcher.visit(layer_func_ast)
    new_layer_func_ast = layer_func_ast

    if matcher.matched:
        return astor.to_source(new_layer_func_ast)
    else:
        return astor.to_source(ast.Module(body=[layer_func_ast.body[0]], type_ignores=[]))

# Example usage
def layer_func_example():
    def sub_func_example():
        pass
    sub_func_example()

def sub_func_example():
    pass

print(match_and_rewrite(layer_func_example, sub_func_example))

# Example usage
def sub_func_example(out, inp, weights):
    for j in range(4):
        out += inp[j] * weights[j]
    return out

rewritten_layer = match_and_rewrite(fully_connected_i8_layer, sub_func_example)
print(rewritten_layer)
def ast_to_c(node):
    if isinstance(node, ast.Module):
        return "\n".join(ast_to_c(stmt) for stmt in node.body)
    elif isinstance(node, ast.FunctionDef):
        args = ", ".join(arg.arg for arg in node.args.args)
        body = "\n".join(ast_to_c(stmt) for stmt in node.body)
        return f"void {node.name}({args}) {{\n{body}\n}}"
    elif isinstance(node, ast.For):
        target = ast_to_c(node.target)
        iter = ast_to_c(node.iter)
        body = "\n".join(ast_to_c(stmt) for stmt in node.body)
        return f"for ({target} = 0; {target} < {iter}; {target}++) {{\n{body}\n}}"
    elif isinstance(node, ast.Assign):
        targets = ", ".join(ast_to_c(t) for t in node.targets)
        value = ast_to_c(node.value)
        return f"{targets} = {value};"
    elif isinstance(node, ast.BinOp):
        left = ast_to_c(node.left)
        right = ast_to_c(node.right)
        op = ast_to_c(node.op)
        return f"{left} {op} {right}"
    elif isinstance(node, ast.Add):
        return "+"
    elif isinstance(node, ast.Sub):
        return "-"
    elif isinstance(node, ast.Mult):
        return "*"
    elif isinstance(node, ast.Div):
        return "/"
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        return str(node.value)
    elif isinstance(node, ast.Expr):
        return ast_to_c(node.value)
    elif isinstance(node, ast.Return):
        value = ast_to_c(node.value)
        return f"return {value};"
    elif isinstance(node, ast.If):
        test = ast_to_c(node.test)
        body = "\n".join(ast_to_c(stmt) for stmt in node.body)
        orelse = "\n".join(ast_to_c(stmt) for stmt in node.orelse)
        return f"if ({test}) {{\n{body}\n}} else {{\n{orelse}\n}}"
    elif isinstance(node, ast.Compare):
        left = ast_to_c(node.left)
        ops = " ".join(ast_to_c(op) for op in node.ops)
        comparators = " ".join(ast_to_c(c) for c in node.comparators)
        return f"{left} {ops} {comparators}"
    elif isinstance(node, ast.Eq):
        return "=="
    elif isinstance(node, ast.Lt):
        return "<"
    elif isinstance(node, ast.Gt):
        return ">"
    elif isinstance(node, ast.LtE):
        return "<="
    elif isinstance(node, ast.GtE):
        return ">="
    elif isinstance(node, ast.NotEq):
        return "!="
    elif isinstance(node, ast.Call):
        func = ast_to_c(node.func)
        args = ", ".join(ast_to_c(arg) for arg in node.args)
        return f"{func}({args})"
    elif isinstance(node, ast.Attribute):
        value = ast_to_c(node.value)
        attr = node.attr
        return f"{value}.{attr}"
    elif isinstance(node, ast.Subscript):
        value = ast_to_c(node.value)
        slice = ast_to_c(node.slice)
        return f"{value}[{slice}]"
    elif isinstance(node, ast.Index):
        return ast_to_c(node.value)
    elif isinstance(node, ast.Slice):
        lower = ast_to_c(node.lower)
        upper = ast_to_c(node.upper)
        step = ast_to_c(node.step)
        return f"{lower}:{upper}:{step}"
    elif isinstance(node, ast.ExtSlice):
        dims = ", ".join(ast_to_c(d) for d in node.dims)
        return f"{dims}"
    elif isinstance(node, ast.NameConstant):
        return str(node.value)
    elif isinstance(node, ast.Pass):
        return "pass"
    elif isinstance(node, ast.IfExp):
        test = ast_to_c(node.test)
        body = ast_to_c(node.body)
        orelse = ast_to_c(node.orelse)
        return f"{body} if {test} else {orelse}"
    elif isinstance(node, ast.UnaryOp):
        operand = ast_to_c(node.operand)
        op = ast_to_c(node.op)
        return f"{op}{operand}"
    elif isinstance(node, ast.USub):
        return "-"
    elif isinstance(node, ast.UAdd):
        return "+"
    elif isinstance(node, ast.Not):
        return "!"
    elif isinstance(node, ast.BoolOp):
        op = ast_to_c(node.op)
        values = ", ".join(ast_to_c(v) for v in node.values)
        return f"{op}({values})"
    elif isinstance(node, ast.And):
        return "&&"
    elif isinstance(node, ast.Or):
        return "||"
    elif isinstance(node, ast.List):
        elts = ", ".join(ast_to_c(elt) for elt in node.elts)
        return f"[{elts}]"
    elif isinstance(node, ast.Tuple):
        elts = ", ".join(ast_to_c(elt) for elt in node.elts)
        return f"({elts})"
    elif isinstance(node, ast.Sub):
        return "-"
    elif isinstance(node, ast.AugAssign):
        target = ast_to_c(node.target)
        op = ast_to_c(node.op)
        value = ast_to_c(node.value)
        return f"{target} {op}= {value}"
    else:
        raise NotImplementedError(f"Unsupported AST node type: {type(node)}")

layer_func_ast = ast.parse(inspect.getsource(fully_connected_i8_layer))
c_code = ast_to_c(layer_func_ast)
print("\nGenerated C code:")
print(c_code)