#ifndef __MATCH_${model_name}_GRAPH_RUNTIME_H__
#define __MATCH_${model_name}_GRAPH_RUNTIME_H__

#include <${model_name}/data.h>

int match_${model_name}_graph_runtime(
    % for inp in inputs:
    ${inp.c_type}* ${inp.name}_pt,
    % endfor
    % for idx,out in enumerate(outputs):
    ${"" if idx==0 else ","}${out.c_type}* ${out.name}_pt
    % endfor
);
#endif