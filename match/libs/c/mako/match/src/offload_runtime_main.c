#ifdef __${exec_module.name}__

#include <stdint.h>

#include "${target.name}.h"
% for include in target.include_list:
    #include <${include}.h>
% endfor


<%def name="primary_core_region()">
% if exec_module.is_smp and platform_apis.smp_primary_core_guard != "":
    if (${platform_apis.smp_primary_core_guard}(NULL)) {
% endif
${caller.body()}
% if exec_module.is_smp and platform_apis.smp_primary_core_guard != "":
    }
% endif
</%def>

<%def name="smp_barrier()">
% if exec_module.is_smp and sync_apis.smp_barrier != "":
    ${sync_apis.smp_barrier}();
% endif
</%def>

<%def name="smp_print()">
% if platform_apis.print_fn != "":
    <%self:primary_core_region>
    ${platform_apis.print_fn}(${caller.body()});
    </%self:primary_core_region>
% endif
</%def>


int main(int argc, char** argv) {
    volatile uint32_t* args = ${mem_apis.shared_memory_extern_addr};

    while (1) {
        <%self:primary_core_region>
            // Polling for the start signal
            while (args[0] == 0) {
                asm volatile("fence r,rw" ::: "memory");
            }
            // Or wait for interrupt
            // asm volatile("wfi");
        </%self:primary_core_region>

        <%self:smp_print>"Mi hanno detto di fare %d + 1\r\n", args[0] - 1</%self:smp_print>
        <%self:smp_print>"Al momento c'è: %p, %p, %p, %p %p, ...\r\n", args[0], args[1], args[2], args[3], args[4]</%self:smp_print>
        
        <%self:smp_barrier/>

        switch (args[0]) {
            case 0:
                break;
            % for node in nodes:
                case ${int(node.fn_name.split("_")[-1])} + 1:
                    ${node.fn_name}_inner(args);
                    break; 
            % endfor
            case ${exec_module.name}_EXIT_SIGNAL:
                // Handle exit signal
                <%self:smp_print>"Exit Signal Received.\r\n"</%self:smp_print>
                return 0;
            default:
                // Handle unknown command
                <%self:smp_print>"Unknown node command: %d\r\n", args[1]</%self:smp_print>
                return -1;
        }

        <%self:smp_barrier/>

        <%self:primary_core_region>
            // Clear the start signal
            asm volatile("fence rw,rw" ::: "memory");
            args[0] = 0;
            asm volatile("fence rw,rw":::"memory");
            // Or send interrupt
            // TODO
        </%self:primary_core_region>
    }

    return 0;
}

#endif