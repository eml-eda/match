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
Operations to support a MATCH Target with one or more ExecModule.
"""

import logging

from match.transform.cast import MatchRemoveFakeOutDtypeCasts
from match.transform.dead import MatchRemoveIdentityBYOC
from match.transform.naming import MatchRenameIO
from match.transform.save import MatchSaveModule, MatchSaveRelay
from match.utils.utils import get_model_name

import tvm
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
# from tvm.driver.tvmc import TVMCException

# don't remove this import even if it does not seem to be used
# because this is the point where the match backend is registered
import tvm.relay.backend.contrib.match

from match.target import get_target, MatchTarget, DefaultMatchTarget

from match.matcha import MatchOptimizer

logger = logging.getLogger("match")


def pattern_table(target: MatchTarget = DefaultMatchTarget()):
    """
    Registers the patterns we want to match.
    Returns
    -------
        The patterns.
    """
    patterns = [
        (f"match.{target_pattern.name}", target_pattern.pattern(), target_pattern.additional_checks)
        for target_pattern in target.partitioning_patterns()
    ]
    return patterns


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
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    target = get_target()

    pipeline = []
    
    def saved_relay_id():
        saved_relay_id.counter += 1
        return saved_relay_id.counter
    saved_relay_id.counter = -1
    
    pipeline.append(MatchSaveRelay(f"{saved_relay_id()}_start"))
    pipeline.append(transform.InferType())
    pipeline.append(MatchRenameIO())
    pipeline.append(MatchSaveRelay(f"{saved_relay_id()}_renamed"))
    
    pipeline.append(transform.InferType())
    # TODO: understand if current broadcast rel of outdtype is fixed with new releases
    # currently its not working completely as expected
    # conv(outdtype="int32") -> multiply() -> add() breaks in TVM when building
    # this doesnt happen with conv(outdtype="int32") -> biasadd
    # now this pass does this --> conv() --> cast(outdtype="int32") -> multiply() -> add()
    # to -->conv(outdtype="int32") --> cast(outdtype="int32") --> multiply() -> add()
    pipeline.append(MatchRemoveFakeOutDtypeCasts())
    pipeline.append(transform.InferType())
    pipeline.append(MatchSaveRelay(f"{saved_relay_id()}_removed_fake_casts"))

    pipeline.append(transform.InferType())
    for net_transform_name, net_transform in target.transform_before_partitioning(opts):
        pipeline.append(net_transform)
        pipeline.append(MatchSaveRelay(f"{saved_relay_id()}_{net_transform_name}"))
        pipeline.append(transform.InferType())
    pipeline.append(transform.FoldConstant())
    pipeline.append(MatchSaveRelay(f"{saved_relay_id()}_folded"))
    
    pipeline.append(transform.InferType())
    pipeline.append(MatchSaveRelay(f"{saved_relay_id()}_transformed"))

    if True:
        pipeline.append(MatchOptimizer(target))
    else:
        pipeline.append(transform.MergeComposite(pattern_table(target=target)))
        
    pipeline.append(transform.AnnotateTarget(["match"]))
    pipeline.append(MatchSaveRelay(f"{saved_relay_id()}_merged"))

    for net_transform_name, net_transform in target.transform_after_partitioning(opts):
        pipeline.append(net_transform)
        pipeline.append(MatchSaveRelay(net_transform_name))
        pipeline.append(transform.InferType())
    pipeline.append(MatchSaveRelay(f"{saved_relay_id()}_adjusted"))
    
    pipeline.append(transform.PartitionGraph(get_model_name(), bind_constants=True))
    pipeline.append(transform.InferType())
    pipeline.append(MatchSaveRelay(f"{saved_relay_id()}_partitioned"))

    pipeline.append(MatchRemoveIdentityBYOC())
    pipeline.append(transform.DeadCodeElimination())
    pipeline.append(transform.RemoveUnusedFunctions())
    pipeline.append(MatchSaveRelay(f"{saved_relay_id()}_cleaned"))

    pipeline.append(MatchSaveModule())
    seq = tvm.transform.Sequential(pipeline)
    with tvm.transform.PassContext(opt_level=3):
        try:
            fused = seq(mod)
            MatchSaveRelay(f"{saved_relay_id()}_final")(fused)
            return fused
        except Exception as exc:
            raise Exception("[PARTITION] Error converting layout to {0}".format(str(exc)))
