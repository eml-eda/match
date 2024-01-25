#ifndef _MATCH_TARGET_PARAMS_H
#define _MATCH_TARGET_PARAMS_H

#define NUM_MEMORY_LEVELS ${max([1]+[len(memory_names[exec_module.name]) for exec_module in exec_modules])}

% for exec_module in exec_modules:
typedef enum{
    % if len(memory_names[exec_module.name])>0:
    ${memory_names[exec_module.name][0]}
    % endif
    % for mem_level in memory_names[exec_module.name][1:]:
    ,${mem_level}
    % endfor
}mem_levels_${exec_module.name};

typedef enum{
    % if len(exec_module.partitioning_patterns())>0:
    ${exec_module.partitioning_patterns()[0].name}
    % endif
    % for pat in exec_module.partitioning_patterns()[1:]:
    ,${pat.name}
    % endfor
}patterns_${exec_module.name};

% if len(exec_module.specific_patterns)==0:
typedef patterns_${exec_module.name} specific_patterns_${exec_module.name};
% else:
typedef enum{
    ${exec_module.specific_patterns[0]}
    % for pat in exec_module.specific_patterns[1:]:
    ,${pat}
    % endfor
}specific_patterns_${exec_module.name};
% endif
% endfor

#endif