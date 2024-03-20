#ifndef _MATCH_TARGET_PARAMS_H
#define _MATCH_TARGET_PARAMS_H

#define NUM_MEMORY_LEVELS ${max([1]+[len(memory_names[exec_module.name]) for exec_module in exec_modules])}

typedef enum{
    % if len(memories_list)>0:
    ${memories_list[0]}
    % endif
    % for mem_level in memories_list[1:]:
    ,${mem_level}
    % endfor
}memories_list;

typedef enum{
    % if len(patterns_list)>0:
    ${patterns_list[0]}
    % endif
    % for pat in patterns_list[1:]:
    ,${pat}
    % endfor
}patterns_list;

#endif