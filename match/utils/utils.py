import os
import pathlib
import json
import subprocess
from typing import Dict, List
from mako.template import Template
import numpy as np
from tvm import relay as relay_tvm
import match
from tvm.ir import IRModule
import tvm

def get_default_inputs(mod: IRModule=None, params: Dict[str, tvm.runtime.ndarray.NDArray]={}, input_files: List[str]=[], min_input_val=None, max_input_val=None):
    default_inputs = []
    if input_files is not None and len(input_files)==len(mod["main"].params):
        default_inputs = [ np.loadtxt(input_files[param_idx], delimiter=',',
                            dtype=np.dtype(param.type_annotation.dtype),
                            usecols=[0]).reshape([int(i) for i in param.type_annotation.shape])
                            for param_idx, param in enumerate(mod["main"].params) if param.name_hint not in params]
    else:
        default_inputs = [get_random_np_array(dtype=param.type_annotation.dtype, shape=param.type_annotation.shape, min_val=min_input_val, max_val=max_input_val)
                           for param in mod["main"].params if param.name_hint not in params]
    return default_inputs

def numpy_dtype_to_c_type(dtype):
    """Translate NumPy dtype to corresponding C type."""
    dtype_str = str(dtype)
    
    # Mapping NumPy dtype to C type
    mapping = {
        'float32': 'float',
        'float64': 'double',
        'int32': 'int',
        'int64': 'long int',  # or 'long long int'
        'int8': 'int8_t',
        'uint8': 'uint8_t',
        'int16': 'short',
        'uint16': 'unsigned short',
        'uint32': 'unsigned int',
        'uint64': 'unsigned long int',  # or 'unsigned long long int'
        'bool': '_Bool'  # or 'bool' in C99
    }
    
    # Check if dtype exists in mapping
    if dtype_str in mapping:
        return mapping[dtype_str]
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def c_friendly_npvalue(arr):
    arr = arr.flatten()
    # params: arr is expected to be a numpy version of the value, it should be an array but it may be also just a single value
    if len(arr.shape)>0:
        # this is actually an array and not a single value
        arr=arr.reshape([arr.shape[0]])
        return f'{{{str(list(arr))[1:len(str(list(arr)))-1]}}}'
    else:
        return str(arr)

def get_random_np_array(dtype, shape, min_val=None, max_val=None):
    shape = [int(i) for i in shape]
    if np.issubdtype(dtype, np.floating):
        if min_val is not None and max_val is not None:
            return (max_val - min_val) * np.random.rand(*shape).astype(dtype) + min_val
        return np.random.rand(*shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.random.randint(info.min if min_val is None or not isinstance(min_val,(int, float)) else min_val,
                                 info.max if max_val is None or not isinstance(max_val,(int, float)) else max_val+1,
                                    size=shape, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

class RelaySave:
    def __init__(self,prefix,mod,params):
        self.prefix=prefix
        self.mod=mod
        self.params=params
        
output_path=None
model_name = "default"
executor = "aot"
relay_list=[]
schedules=[]
searched_schedules=[]

fname_to_node_schedule = {}

def reset_schedules():
    global schedules
    global searched_schedules
    schedules=[]
    searched_schedules=[]

def reset_relay_list():
    global relay_list
    relay_list=[]

def reset_output_path():
    global output_path
    output_path=None

def set_output_path(path):
    global output_path
    output_path=path

def get_output_path():
    global output_path
    return output_path

def set_model_name(mod_name):
    global model_name
    model_name = mod_name

def get_model_name():
    global model_name
    return model_name

def set_executor(exec_):
    global executor
    executor = exec_

def get_executor():
    global executor
    return executor

def add_fname_node_schedule(fname, node, schedule, node_name, cpu_only_c_lib, cpu_only_llvm_lib):
    global fname_to_node_schedule
    fname_to_node_schedule[fname] = (node, schedule, node_name, cpu_only_c_lib, cpu_only_llvm_lib)

def get_fname_node_schedule(fname):
    global fname_to_node_schedule
    return fname_to_node_schedule[fname]

def mock_func(*args):
    return None

def get_x86_result(output_path: str, verbose: bool = False, keep_result: bool = False):
    output={"output":[-1,-1,-1,-1]}
    print("Building ...")

    with open(output_path/"x86_output.json","wb") as logfile:
        output1 = subprocess.Popen([pathlib.Path(os.path.dirname(__file__)+"/x86_lib/run_x86_match.sh"),output_path,pathlib.Path(os.path.dirname(__file__)+"/x86_lib")],stdout=subprocess.PIPE)
        output2 = subprocess.Popen(["grep","]}"],stdin=output1.stdout,stdout=logfile)
        output2.wait()
    with open(output_path/"x86_output.json") as logfile:
        output=json.load(logfile)

    if not keep_result:
        subprocess.run(["rm",output_path/"x86_output.json"])
    return output

def x86_run_match(input_type="onnx",relay_mod=None, relay_params=None, filename=None, params_filename=None, output_path="./tmp/x86_test",keep_result: bool = False):
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    x86_res=match.match(input_type=input_type,relay_mod=relay_mod,relay_params=relay_params,filename=filename,params_filename=params_filename,
                    target_name="",output_path=pathlib.Path(output_path))
    main_code_template=Template(filename=os.path.dirname(__file__)+"/x86_lib/x86_main_template.c")
    template_data_dict=x86_res.__dict__
    template_data_dict["target"]="x86"
    main_code=main_code_template.render(**template_data_dict)
    with open(pathlib.Path(output_path)/"src/main.c","w") as main_file:
        main_file.write(main_code)

    x86_result=get_x86_result(pathlib.Path(output_path),keep_result=keep_result)
    return x86_result

def add_save_relay(prefix="",mod=None,params=None):
    global relay_list
    relay_list.append(RelaySave(
        prefix=prefix,
        mod=None if mod is None else relay_tvm.astext(mod),
        params=None if params is None else relay_tvm.save_param_dict(params=params)
    ))
    save_all_relay()

def save_all_relay():
    global relay_list
    for relay_save in relay_list:
        if relay_save.mod is not None:
            with open(f"{get_output_path()}/{relay_save.prefix}_graph.relay","w") as mod_file:
                mod_file.write(relay_save.mod)
        if relay_save.params is not None:
            with open(f"{get_output_path()}/{relay_save.prefix}_params.txt","wb") as par_file:
                par_file.write(relay_save.params)

def save_codegen_schedule(node,schedule,latency,energy):
    loops_str=""
    for idx_block,block in enumerate(schedule.blocks):
        loops_str+=f"Block {idx_block}\n"
        for idx,lp in enumerate(block.loops):
            loops_str+="\t"*idx
            op_str=""
            for mem_transfer in lp.mem_transfers:
                op_str+=f" [load tensor {mem_transfer.tensor.name}] "
            loops_str+=f"for {lp.name} in 0:{lp.size}: {op_str}\n"

    schedules.append(f"\nFor node\n{node}\
                    \nMATCH found that the best schedule, with expected latency {latency} and energy {energy}, has the following temporal mapping\
                    \n{loops_str}\n")

def save_schedule_search_res(name,latency,energy,schedule,node):
    loops_str=""
    for idx_block,block in enumerate(schedule.blocks):
        loops_str+=f"Block {idx_block}\n"
        for idx,lp in enumerate(block.loops):
            loops_str+="\t"*idx
            op_str=""
            for mem_transfer in lp.mem_transfers:
                op_str+=f" [load tensor {mem_transfer.tensor.name}] "
            loops_str+=f"for {lp.name} in 0:{lp.size}: {op_str}\n"

    searched_schedules.append(f"\nFor node\n{node}\
                    \nMATCH found a schedule with pattern {name} with expected latency {latency} and energy {energy}\n with the following temporal mapping\
                    \n{loops_str}\n")

# def save_schedule_search_res(match_node,schedule,latency,energy):
#     tmap_searched.append(f"\nFor match node: {match_node} schedule:\
#                     \n{schedule}\nhas expected latency {latency} and energy {energy}\n")

def save_all_schedules():
    with open(f"{get_output_path()}/schedules.log","w") as scheds_file:
        scheds_file.writelines(schedules)
    with open(f"{get_output_path()}/searched_schedules.log","w") as scheds_file:
        scheds_file.writelines(searched_schedules)
    # with open(f"{get_output_path()}/match_searched_tmaps.log","w") as scheds_file:
    #     scheds_file.writelines(tmap_searched)