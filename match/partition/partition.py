
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Operations to support the SOMA accelerator.
"""

from typing import Any, List
import tvm
import logging
from functools import partial
from tvm import relay

from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
#from tvm.driver.tvmc import TVMCException

# don't remove this import even if it does not seem to be used
# because this is the point where the match backend is registered
import tvm.relay.backend.contrib.match
from match.target.target import MatchTarget

#from match.matchutils import get_target
from match.target import get_target
from tvm.relay.dataflow_pattern import DFPatternCallback, is_op, rewrite, wildcard

logger = logging.getLogger("match")

def pattern_table(target:MatchTarget=None):
    """
    Registers the patterns we want to match.
    Returns
    -------
        The patterns.
    """
    patterns=[(
        f"match.{target_pattern.name}", 
        target_pattern.pattern(), 
        target_pattern.additional_checks) 
        for target_pattern in target.partitioning_patterns()]
    return patterns


class RewriteOnnxBiasesCallback(DFPatternCallback):
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.conv2d = is_op("nn.conv2d")(wildcard(),wildcard())
        self.bias = wildcard()
        self.pattern = is_op("nn.bias_add")(self.conv2d,self.bias)

    def callback(self, pre, post, node_map):
        conv2d = node_map[self.conv2d][0]
        out_dtype=conv2d.attrs["out_dtype"]
        bias = node_map[self.bias][0]
        return relay.op.nn.bias_add(relay.op.cast(conv2d, out_dtype),bias)


@tvm.ir.transform.module_pass(opt_level=0)
class RewriteOnnxBiases:
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        global_var=mod.get_global_var("main")
        func=mod.functions[global_var]
        func = rewrite(RewriteOnnxBiasesCallback(), func)
        mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)

def partition(mod, params, dpu, opts):
    """
    The partitioning sequence for the MATCH byoc
    Parameters
    ----------
    mod The module to use

    Returns
    -------
    The partitioned module.

    """
    #breakpoint()
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    target_name="" if "target" not in opts else opts["target"]
    
    target=get_target(target_name=target_name)

    pipeline = []

    pipeline+=target.adjust_network(opts)

    pipeline.append(transform.InferType())
    pipeline.append(transform.MergeComposite(pattern_table(target=target)))
    pipeline.append(transform.AnnotateTarget(["match"]))

    pipeline+=target.network_transformations(opts)

    pipeline.append(transform.InferType())
    pipeline.append(transform.PartitionGraph())
    pipeline.append(transform.InferType())

    pipeline.append(RewriteOnnxBiases())
    pipeline.append(transform.InferType())
    seq = tvm.transform.Sequential(pipeline)
    with tvm.transform.PassContext(opt_level=3):
        try:
            fused = seq(mod)
            return fused
        except Exception as err:
            raise Exception(
                "Error converting layout to {0}".format(str(err))
            )
