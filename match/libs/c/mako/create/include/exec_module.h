#ifndef __MATCH_EXEC_MODULE_${exec_module.name}_H__
#define __MATCH_EXEC_MODULE_${exec_module.name}_H__

typedef enum{
    % for mem_level in exec_module.memories:
    ${mem_level},
    % endfor
}${exec_module.name}_memories;

typedef enum{
    default_pattern
    % for pat in exec_module.partitioning_patterns():
    ,${pat.name}
    % endfor
}${exec_module.name}_patterns;

% for module_option_name,module_option_value in exec_module.module_options.items():
#define ${module_option_name} ${module_option_value}
% endfor

## // MEM APIS
## % for mem_api_name,mem_api_mapping in exec_module.mem_apis:
## #define __${mem_api_name}__ ${mem_api_mapping}
## % endfor

## // COMP APIS
## % for comp_api_name,comp_api_mapping in exec_module.comp_apis:
## #define __${comp_api_name}__ ${comp_api_mapping}
## % endfor

## // SYNC APIS
## % for sync_api_name,sync_api_mapping in exec_module.sync_apis:
## #define __${sync_api_name}__ ${sync_api_mapping}
## % endfor

## // PLATFORM APIS
## % for platform_api_name,platform_api_mapping in exec_module.platform_apis:
## #define __${platform_api_name}__ ${platform_api_mapping}
## % endfor

#endif