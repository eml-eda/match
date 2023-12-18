% for exec_module in exec_modules:
#define NUM_MEMORY_LEVELS_${exec_module.name} ${len(memory_names[exec_module.name])}
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
% endfor