#ifndef __MATCH_TARGET_${target.name}_H__
#define __MATCH_TARGET_${target.name}_H__
% for ex_mod in target.exec_modules:
#include <${ex_mod.name}.h>
% endfor

#define __${target.name.upper()}__NUM_EXEC_MODULE__ ${len(target.exec_modules)}

% for mem_idx,memory in enumerate(target.host_memories()[::-1]):
#ifndef __${memory.name.upper()}__
#define __${memory.name.upper()}__
#define ${memory.name} ${mem_idx}
#endif
% endfor

% for exec_module in target.exec_modules:
% for mem_idx,memory in enumerate(exec_module.module_memories()[::-1]):
#ifndef __${memory.name.upper()}__
#define __${memory.name.upper()}__
#define ${memory.name} ${mem_idx+len(target.host_memories())}
#endif
% endfor
% endfor

% for pat_idx, pat in enumerate([pt for exec_module in target.exec_modules for pt in exec_module.partitioning_patterns()]):
#ifndef __${pat.name.upper()}__
#define __${pat.name.upper()}__
#define ${pat.name} ${pat_idx}
#endif
% endfor

#endif