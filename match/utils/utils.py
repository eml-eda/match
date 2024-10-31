import os
import pathlib
import json
import subprocess
from mako.template import Template
from tvm import relay as relay_tvm
import match

class RelaySave:
    def __init__(self,prefix,mod,params):
        self.prefix=prefix
        self.mod=mod
        self.params=params
        
output_path=None
relay_list=[]
schedules=[]
searched_schedules=[]
tmap_searched=[]

def reset_schedules():
    global schedules
    global searched_schedules
    global tmap_searched
    schedules=[]
    searched_schedules=[]
    tmap_searched=[]

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
    # FOR TESTING PURPOSES
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

def save_codegen_schedule(node,temporal_mapping,spatial_mapping,latency,energy):
    loops_str=""
    for idx,tmap in enumerate(temporal_mapping):
        loops_str+="\t"*idx
        ops=[k[4:] for k in tmap.keys() if 'mem_' in k]
        op_str=""
        for op in ops:
            mem_op=f'mem_{op}'
            op_str+=f" [{op} in {tmap[mem_op]}] "
        loops_str+=f"for {tmap['fullname']} in 0:{tmap['size']}: {op_str}\n"

    schedules.append(f"\nFor node\n{node}\
                    \nMATCH found that the best schedule, with expected latency {latency} and energy {energy}, has the following temporal mapping\
                    \n{loops_str}\n")

def save_schedule_search_res(name,latency,energy,temporal_mapping,node):
    loops_str=""
    for idx,tmap in enumerate(temporal_mapping):
        loops_str+="\t"*idx
        ops=[k[4:] for k in tmap.keys() if 'mem_' in k]
        op_str=""
        for op in ops:
            mem_op=f'mem_{op}'
            op_str+=f" [{op} in {tmap[mem_op]}] "
        loops_str+=f"for {tmap['fullname']} in 0:{tmap['size']}: {op_str}\n"

    searched_schedules.append(f"\nFor node\n{node}\
                    \nMATCH found a schedule with pattern {name} with expected latency {latency} and energy {energy}\n with the following temporal mapping\
                    \n{loops_str}\n")
    tmap_searched.append("\n#################------##############\n")

def save_tmap_search_res(layer_data,temporal_mapping,latency,energy):
    tmap_searched.append(f"\nFor layer data with sizes: {layer_data.loop_dim_size} and strides {layer_data.strides} tmap:\
                    \n{temporal_mapping}\nhas expected latency {latency} and energy {energy}\n")

def save_all_schedules():
    with open(f"{get_output_path()}/match_schedules.log","w") as scheds_file:
        scheds_file.writelines(schedules)
    with open(f"{get_output_path()}/match_searched_schedules.log","w") as scheds_file:
        scheds_file.writelines(searched_schedules)
    with open(f"{get_output_path()}/match_searched_tmaps.log","w") as scheds_file:
        scheds_file.writelines(tmap_searched)