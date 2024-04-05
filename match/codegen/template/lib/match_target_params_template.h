#ifndef _MATCH_TARGET_PARAMS_H
#define _MATCH_TARGET_PARAMS_H

#define NUM_MEMORY_LEVELS ${max([1]+[len(memory_names[exec_module.name]) for exec_module in exec_modules])}

% if len(memories_list)>0:
typedef enum{
    ${memories_list[0]}
    % for mem_level in memories_list[1:]:
    ,${mem_level}
    % endfor
}memories_list;
% endif

% if len(patterns_list)>0:
typedef enum{
    ${patterns_list[0]}
    % for pat in patterns_list[1:]:
    ,${pat}
    % endfor
}patterns_list;
% endif

% for exec_module in exec_modules:
% for module_option_name,module_option_value in exec_module.module_options.items():
#define ${module_option_name} ${module_option_value}
% endfor
% endfor

#endif