import os
import pathlib
from subprocess import Popen,PIPE
import json
import subprocess
from mako.template import Template

import match

def mock_func(*args):
    return None

def get_x86_result(output_path: str, verbose: bool = False):
    output={"output":[-1,-1,-1,-1]}
    print("Building ...")
    subprocess.run(["./run_x86_match.sh",output_path],cwd=os.path.dirname(__file__)+"/x86_lib")

    with open(output_path/"x86_output.json") as logfile:
        output=json.load(logfile)

    Popen(["rm",output_path/"x86_output.json"])
    return output

def x86_run_match(input_type="onnx",relay_mod=None, relay_params=None, filename=None, params_filename=None, output_path="./tmp/x86_test"):
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    x86_res=match.match(input_type=input_type,relay_mod=relay_mod,relay_params=relay_params,filename=filename,params_filename=params_filename,
                    target_name="",output_path=pathlib.Path(output_path))
    main_code_template=Template(filename=os.path.dirname(__file__)+"/x86_lib/x86_main_template.c")
    template_data_dict=x86_res.__dict__
    template_data_dict["target"]="x86"
    main_code=main_code_template.render(**template_data_dict)
    with open(pathlib.Path(output_path)/"src/main.c","w") as main_file:
        main_file.write(main_code)

    x86_result=get_x86_result(pathlib.Path(output_path))
    return x86_result