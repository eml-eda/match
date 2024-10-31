#breakpoint()


import os
import subprocess
import tvm
from tvm import relax
from tvm.script.parser import relax as R
import numpy as np
from tvm.relax.dpl import is_op, wildcard
from tvm.relax import ExprFunctor
from tvm.relax.expr import (
    Binding,
    BindingBlock,
    Call,
    Constant,
    Id,
    DataflowBlock,
    DataflowVar,
    DataTypeImm,
    Expr,
    ExternFunc,
    Function,
    GlobalVar,
    If,
    MatchCast,
    PrimValue,
    SeqExpr,
    ShapeExpr,
    Span,
    StringImm,
    Tuple,
    TupleGetItem,
    Var,
    VarBinding,
)
from tvm.ir import Op
ONNX_MOD =  False
SIMPLE_ADD = True

class PrintEl:
    def __init__(self, data_type: str="char*", deref: str="", print_type: str="%c", index: str="") -> None:
        self.data_type=data_type
        self.deref=deref
        self.print_type=print_type
        self.index=index

class RelaxGraphTraverser(ExprFunctor):
    def __init__(self):
        super().__init__()
        self.functions = []
        self.variables = []
        self.calls = []
        self.tuples = []

    def visit_constant_(self, op: Constant):
        print("const...")

    def visit_tuple_(self, op: Tuple):
        print("tuple...")

    def visit_dataflow_var_(self, op: DataflowVar):
        print("dataflow var...")

    def visit_var_(self, op: Var):
        print("var...")
        breakpoint()

    def visit_shape_expr_(self, op: ShapeExpr):
        print("shape expr...")

    def visit_extern_func_(self, op: ExternFunc):
        print("extern func...")

    def visit_global_var_(self, op: GlobalVar):
        print("global var...")

    def visit_function_(self, op: Function):
        print("func...")
        for param in op.params:
            self.visit_expr(param)
        self.visit_expr(op.body)

    def visit_call_(self, op: Call):
        print("call...")
        for arg in op.args:
            self.visit_expr(arg)

    def visit_seq_expr_(self, op: SeqExpr):
        print("seq expr...")
        for binding_block in op.blocks:
            self.visit_binding_block(binding_block)
        self.visit_expr(op.body)

    def visit_if_(self, op: If):
        print("if...")

    def visit_op_(self, op: Op):
        print("op...")

    def visit_tuple_getitem_(self, op: TupleGetItem):
        print("tuple get item...")

    def visit_prim_value_(self, op: PrimValue):
        print("prim val...")

    def visit_string_imm_(self, op: StringImm):
        print("string imm...")

    def visit_data_type_imm_(self, op: DataTypeImm):
        print("data type imm...")

    def visit_var_binding_(self, binding: VarBinding):
        print("var binding...")
        self.visit_expr(binding.value)

    def visit_match_cast_(self, binding: MatchCast):
        print("match cast...")

    def visit_binding_block_(self, block: BindingBlock):
        print("binding block...")
        for binding in block.bindings:
            self.visit_binding(binding)

    def visit_dataflow_block_(self, block: DataflowBlock):
        print("dataflow block...")

    def visit_var_def_(self, var: Var):
        print("var def...")

    def visit_dataflow_var_def_(self, var: DataflowVar):
        print("dataflow var def...")

# Example: Starting with a Relax function
def traverse_relax_function(func):
    traverser = RelaxGraphTraverser()
    traverser.visit_expr(func)
    return traverser

@tvm._ffi.register_func("relax.ext.matchest")
def matchest_compiler(funcs, options, constant_names):
    """
    Create a CoreML runtime from a Relax module.
    """
    def gen_code():
        breakpoint()
        compiled_funcs = []
        c_code_funcs = []
        for func in funcs:
            name=func.attrs.global_symbol
            func_params = [
                "input_tensor1",
                "input_tensor2",
                "output_tensor",
            ]
            func_params_signature="("+" ".join([("," if idx>0 else "")+"void* "+param_name for idx,param_name in enumerate(func_params)])+")"
            loops = f"   for(int h_=0;h_<16;h_++) for(int w_=0;w_<16;w_++) *((uint8_t*){func_params[2]})+=((uint8_t*){func_params[0]})[h_*16+w_]+((uint8_t*){func_params[1]})[h_*16+w_];\n"
            prints_to_do = {
                "input_tensor1":PrintEl(
                    data_type="int",
                    deref="",
                    print_type="%d",
                    #index="[0]"
                ),
                "input_tensor2":PrintEl(
                    data_type="int",
                    deref="",
                    print_type="%d",
                    #index="[0]"
                ),
                "output_tensor":PrintEl(
                    data_type="int",
                    deref="",
                    print_type="%d",
                    #index="[0]"
                ),
            }
            include_list = [
                "stdio",
                "stdlib",
                "stdint",
                #"types",
            ]
            include_list_prints = [f"#include <{inc_lib}.h>\n" for inc_lib in include_list]
            func_ret_type = "int"
            prints = ['   printf("\\n'+par_name+attrs.index+" "+attrs.print_type+'\\n",'+attrs.deref+"(("+attrs.data_type+")"+par_name+")"+attrs.index+");\n" for par_name,attrs in prints_to_do.items()]
            c_code_func="\n".join(include_list_prints)+func_ret_type+" "+name+func_params_signature+"{\n"+"\n".join(prints)+loops+"}\n"
            c_code_funcs.append(c_code_func)

        model_dir = os.getcwd() + "/tmp/"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        
        c_code = "\n".join(c_code_funcs)
        out_name="matchest_test_0"

        with open(model_dir+out_name+".c", "w") as f:
            f.write(c_code)

    model_dir = os.getcwd() + "/tmp/"
    out_name="gpt_test"
    breakpoint()
    traverse_relax_function(funcs[0])
    tvm_includes = ["-I/home/moyne/phd/compilers/llm-relax/tvm/include","-I/home/moyne/phd/compilers/llm-relax/tvm/3rdparty/dlpack/include"]
    subprocess.run(["gcc", "-shared", "-o", model_dir+out_name+".so", model_dir+out_name+".c", "-fPIC"]+tvm_includes)
    # Clean up the intermediate C file if desired
    #os.remove(model_dir+out_name+".c")
    return [tvm.runtime.load_module(model_dir+out_name+".so")]


#breakpoint()

# Define an example IRModule
@tvm.script.ir_module
class InputModule:
    @R.function
    def main(
        x: R.Tensor((1,"width"), "float32"), y: R.Tensor((1,"width"), "float32")
    ) -> R.Tensor((1,"width"), "float32"):
        with R.dataflow():
            # Simple add
            z4 = R.add(x,y)
            z5 = R.multiply(z4,x)
            R.output(z5)
            # More complex
            #z1 = R.add(x, y)
            #z2 = R.add(z1, x)
            #z3 = R.add(z1, z2)
            #z4 = R.add(z3, z2)
            #z5 = R.add(z4, z1)
            #R.output(z5)
        return z5

if ONNX_MOD:
    import onnx
    from tvm import relay
    from tvm.relax.frontend.onnx import from_onnx

    onnx_model_path = '/home/moyne/phd/compilers/match/examples/decoder_model.onnx'
    onnx_model = onnx.load(onnx_model_path)

    # Convert ONNX model to Relay
    shape_dict = {}  # Provide input shapes here
    relay_mod, params = from_onnx(onnx_model)

    mod = relay_to_relax(relay_mod, params)
else:
    mod = InputModule

def add_mul_pt():
    var_a = wildcard()
    return is_op("relax.multiply")(is_op("relax.add")(var_a,wildcard()),var_a)

patterns = [
    ("matchest.cpu_addmul", add_mul_pt()),
    ("matchest.cpu_multiply", is_op("relax.multiply")(wildcard(), wildcard())),
    ("matchest.cpu_add", is_op("relax.add")(wildcard(), wildcard())),
]
mod1 = relax.transform.FuseOpsByPattern(patterns)(mod)

mod2 = relax.transform.MergeCompositeFunctions()(mod1)

mod3 = relax.transform.RunCodegen()(mod2)

# Produced runtime module will be attached in the IRModule attribute.
#print(f"TensorRT runtime module: {mod3.attrs['external_mods']}")


# Check if output IRModule is well-formed. 
assert relax.analysis.well_formed(mod3)

# Define your target hardware and device.
target = tvm.target.Target("llvm")

# Prepare inputs.
np0 = np.random.rand(1, 16).astype(np.float32)
np1 = np.random.rand(1, 16).astype(np.float32)
data0 = tvm.nd.array(np0)
data1 = tvm.nd.array(np1)
inputs = [data0, data1]
print(inputs)

# Prepare expected output.
if SIMPLE_ADD:
    expected = np.add(np0,np1)
else:
    t1 = np.multiply(np0, np1)
    t2 = np.add(t1, np0)
    t3 = np.add(t1, t2)
    t4 = np.multiply(t3, t2)
    expected = np.add(t4, t1)

# Build and prepare VM. 
ex = relax.build(mod3, target, params={},
                 exec_mode="compiled"
                 )
dev = tvm.cpu()
vm = relax.VirtualMachine(ex,dev,profile=True)

# Run VM. 
out = vm.profile("main",*inputs)
print(out)
breakpoint()
ex.export_library("libaddexport.so",fpack_imports=lambda self,is_system_lib, pack_lib_prefix, workspace_dir:"/home/moyne/phd/compilers/match/relax/tmp/gpt_test.o")
import tvm.testing
tvm.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)