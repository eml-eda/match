
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

import tvm
import logging
from functools import partial

from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
#from tvm.driver.tvmc import TVMCException

from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant, is_expr

# don't remove this import even if it does not seem to be used
# because this is the point where the match backend is registered
import tvm.relay.backend.contrib.match
from tvm.relay.backend.contrib.match.transform import MatchOnnxRequantTransform, MatchOnnxIntegerize


logger = logging.getLogger("Match")


def _requant_pattern(prev_op):
    """Add requant pattern (right_shift -> clip -> cast) to prev_op"""
    right_shift = is_op("right_shift")(prev_op, is_constant())
    clip = is_op("clip")(right_shift)
    cast = is_op("cast")(clip).has_attr({"dtype": "uint8"})
    return cast


def _biasadd_requant_pattern(linear_op):
    """Add pattern bias_add-requant to linear_op"""

    bias_add = is_op("nn.bias_add")(linear_op, wildcard()) | is_op("add")(linear_op, wildcard())
    return _requant_pattern(bias_add)

def _bn_requant_pattern(linear_op):
    """Add pattern bias_add-requant to linear_op"""

    bn_mul = is_op("multiply")(linear_op, wildcard())
    bn_add = is_op("add")(bn_mul, wildcard())
    return _requant_pattern(bn_add)



def conv2d_pattern():
    """Create pattern for conv2D with optional fused relu."""

    conv2d_bias = _biasadd_requant_pattern(
        is_op("nn.conv2d")(
            wildcard(), wildcard()
        )
    )
    conv2d_batchnorm = _bn_requant_pattern(
        is_op("nn.conv2d")(
            wildcard(), wildcard()
        )
    )

    return conv2d_bias  | conv2d_batchnorm 


def fully_connected_pattern():
    """Create pattern for nn.dense with optional fused relu."""

    fc = is_op("nn.dense")(
        wildcard(), wildcard()
    )
    return _biasadd_requant_pattern(fc)


def element_wise_add_pattern():
    """Create pattern for element-wise-add with optional fused relu."""

    cast_a = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
    cast_b = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
    add = is_op("add")(cast_a, cast_b)
    return _requant_pattern(add)


def _check_requant(pattern):
    """Check if requant pattern is supported by the gap9 accelerator
    Returns None if not supported, returns the op before this sequence if supported
    """
    return True

def _check_biasadd_or_batchnorm_requant(pattern):
    """Check if pattern is supported by the gap9 accelerator
    Returns None if not supported, returns the linear op before this sequence if supported
    """
    return True



def _check_biasadd_requant(pattern):
    """Check if bias_add-requant pattern is supported by the gap9 accelerator
    Returns None if not supported, returns the linear op before this sequence if supported
    """

    return True


def check_conv2d(pattern,device="gap9"):
    # TODO: FIX THIS, probably needs two different functions for checking? 
    """Check if the Conv2D is supported by the gap9 accelerator"""
    return True


def check_fully_connected(pattern,device="gap9"):
    """Check if the fully connected layer is supported by the gap9 accelerator"""
    return True


def check_element_wise_add(pattern, device="gap9"):
    """Check if the element-wise-add layer is supported by gap9"""
    return True


def pattern_table(opts):
    """
    Registers the patterns we want to match.
    Returns
    -------
        The patterns.
    """
    pt_table=True
    if not pt_table:
        return []
    else:
        return [
            ("match.conv2d", conv2d_pattern(), partial(check_conv2d,device=opts["device"])),
            ("match.dense", fully_connected_pattern(), partial(check_fully_connected,device=opts["device"])),
            ("match.add", element_wise_add_pattern(), partial(check_element_wise_add, device=opts["device"])),
        ]


def partition(mod, params, dpu, opts):
    """
    The partitioning sequence for the match byoc
    Parameters
    ----------
    mod The module to use

    Returns
    -------
    The partitioned module.

    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    pipeline = []
    if 'requant_transform' not in opts or opts['requant_transform'] != '0':
        pipeline.append(MatchOnnxRequantTransform())

   
    pipeline.append(MatchOnnxIntegerize('uint8'))
    pipeline.append(transform.InferType())
    pipeline.append(transform.MergeComposite(pattern_table(opts)))
    pipeline.append(transform.AnnotateTarget(["match"]))
    pipeline.append(transform.InferType())
    pipeline.append(transform.PartitionGraph(mod_name=opts["device"]))
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
