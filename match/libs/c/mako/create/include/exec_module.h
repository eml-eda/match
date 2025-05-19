#ifndef __MATCH_EXEC_MODULE_${exec_module.name}_H__
#define __MATCH_EXEC_MODULE_${exec_module.name}_H__

#include <${target.name}.h>

#define ${exec_module.name.upper()} ${exec_module_id}

% for module_option_name,module_option_value in exec_module.module_options.items():
#define ${module_option_name} ${module_option_value}
% endfor

% if exec_module.separate_build:
    #define ${exec_module.name}_EXIT_SIGNAL 0xFFFFFFFF
%endif

#endif