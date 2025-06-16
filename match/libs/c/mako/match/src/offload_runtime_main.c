#ifdef __${exec_module.name}__

#include <stdint.h>

#include "${target.name}.h"
% for include in target.include_list:
    #include <${include}.h>
% endfor
% for include in exec_module.include_list():
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
    ${sync_apis.smp_barrier}(NULL);
% endif
</%def>

<%def name="smp_print()">
% if platform_apis.print_fn != "":
    <%self:primary_core_region>
    ${platform_apis.print_fn}(${caller.body()});
    </%self:primary_core_region>
% endif
</%def>


static volatile uint32_t* tensor_ptrs;
static volatile int32_t   task_id;

int main(int argc, char** argv) {

    while (1) {
        <%self:primary_core_region>
            ${platform_apis.wait_for_task_fn}(&tensor_ptrs, &task_id);
        </%self:primary_core_region>
        
        <%self:smp_barrier/>

        switch (task_id) {
            % for node in nodes:
                case ${int(node.fn_name.split("_")[-1])}:
                    ${node.fn_name}_inner(tensor_ptrs);
                    break; 
            % endfor
            case ${exec_module.name}_EXIT_SIGNAL:
                // Handle exit signal
                <%self:smp_print>"Exit Signal Received.\r\n"</%self:smp_print>
                return 0;
            default:
                // Handle unknown command
                <%self:smp_print>"Unknown node command: %d\r\n", task_id</%self:smp_print>
                return -1;
        }

        <%self:smp_barrier/>

        <%self:primary_core_region>
            ${platform_apis.end_of_task_fn}(task_id);
        </%self:primary_core_region>
    }

    return 0;
}

#endif