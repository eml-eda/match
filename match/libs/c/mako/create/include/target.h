#ifndef __MATCH_TARGET_${target.name}_H__
#define __MATCH_TARGET_${target.name}_H__
% for ex_mod in target.exec_modules:
#include <${ex_mod.name}/${ex_mod.name}.h>
% endfor

#endif