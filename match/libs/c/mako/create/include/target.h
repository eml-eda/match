#ifndef __MATCH_TARGET_${target.name}_H__
#define __MATCH_TARGET_${target.name}_H__

#include <stdint.h>

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
    <%
        # A tensor's tiles / pts arrays have one entry per memory of the hierarchy of its
        # tensor type (Target.memory_hierarchy_for_pt), so a memory index is its level in
        # the hierarchy of the tensor types it serves, not its position among all memories
        module_memories = exec_module.module_memories()
        def mem_level(memory):
            return max(
                len(target.host_memories()) + [m for m in module_memories if t_type in m.tensor_types][::-1].index(memory)
                for t_type in memory.tensor_types
            )
    %>
    % for memory in module_memories[::-1]:
        #ifndef __${memory.name.upper()}__
        #define __${memory.name.upper()}__
        #define ${memory.name} ${mem_level(memory)}
        #endif
    % endfor

    // extern volatile uint32_t ${exec_module.name}_args[16];
% endfor

% for pat_idx, pat in enumerate([pt for exec_module in target.exec_modules for pt in exec_module.partitioning_patterns()]):
#ifndef __${pat.name.upper()}__
#define __${pat.name.upper()}__
#define ${pat.name} ${pat_idx}
#endif
% endfor

#endif