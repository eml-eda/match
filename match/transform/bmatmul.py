# MatchBatchMatmulTranspose module pass for TVM Relay
# Ensures all nn.batch_matmul nodes have transpose_a=False and transpose_b=False.
# If transpose_a or transpose_b are true, inserts a relay.transpose on the corresponding input
# that swaps the last two dimensions, then calls batch_matmul with both transpose flags set to False.
#
# Usage:
#   from match_batch_matmul_transpose import MatchBatchMatmulTranspose
#   mod2 = MatchBatchMatmulTranspose()(mod)    # apply pass directly (decorated module_pass)
#
# Notes:
# - This pass runs type-inference (InferType) on the module to determine the rank of tensors,
#   so it can create a transpose that swaps the last two axes. It falls back to assuming 3-D
#   (axes=(0,2,1)) when rank information is not available.
# - The pass only touches ops named "nn.batch_matmul".
# - It preserves everything else in the module.

import tvm
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.ty import TensorType


class _BatchMatmulTransposeMutator(ExprMutator):
    """
    ExprMutator that rewrites nn.batch_matmul calls to ensure transpose_a and transpose_b are False.
    If either transpose flag was True, inserts relay.transpose that swaps the last two axes
    of the corresponding operand.
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def _swap_last_two_axes(self, expr):
        """
        Return relay.transpose(expr, axes=...) where axes swaps the last two dimensions.
        Tries to use static rank information from checked_type; if not available falls back to 3-D axes (0,2,1).
        """
        ty = getattr(expr, "checked_type", None)
        rank = None
        if isinstance(ty, TensorType):
            try:
                rank = len(ty.concrete_shape)
            except Exception:
                # concrete_shape may raise if shape contains tvm.tir.Any; fallback to length of ty.shape
                rank = len(ty.shape)
        if rank is None or rank < 2:
            # No reliable rank info: assume 3-D and swap last two dims -> (0,2,1)
            axes = [0, 2, 1]
        else:
            # build axes list that keeps first (rank-2) dims as is, then swaps last two
            axes = list(range(0, max(0, rank - 2))) + [rank - 1, rank - 2]
        return relay.transpose(expr, axes=axes)

    def visit_call(self, call):
        # recurse first
        new_call = super().visit_call(call)

        # Check this is nn.batch_matmul
        op = new_call.op
        if isinstance(op, tvm.ir.Op) and getattr(op, "name", "") == "nn.batch_matmul":
            attrs = new_call.attrs
            # Some attrs can be IntImm or bool-like; coerce to bool
            transpose_a = bool(getattr(attrs, "transpose_a", False))
            transpose_b = bool(getattr(attrs, "transpose_b", False))

            if not (transpose_a or transpose_b):
                return new_call  # already fine

            # Prepare new operands: insert transpose on operands whose transpose flag was true
            lhs, rhs = new_call.args

            if transpose_a:
                lhs = self._swap_last_two_axes(lhs)
            if transpose_b:
                rhs = self._swap_last_two_axes(rhs)

            # Return new batch_matmul with both transpose flags set to False
            # Preserve any attributes other than transpose flags if present (none are typical)
            attrs_dict = {key: attrs[key] for key in attrs.keys()}
            attrs_dict["transpose_a"] = False
            attrs_dict["transpose_b"] = False
            return relay.nn.batch_matmul(lhs, rhs, **attrs_dict)

        return new_call


@tvm.ir.transform.module_pass(opt_level=0)
class BatchMatmulTranspose:
    """
    Module pass that ensures all nn.batch_matmul ops have transpose_a=False and transpose_b=False.
    If transpose flags were true, inserts transpose operations before the batch_matmul that swap
    the last two dimensions of the corresponding input tensors.
    """

    def transform_module(self, mod, ctx):
        # First, run type inference so we can look up static ranks (checked_type) for inputs
        mod = relay.transform.InferType()(mod)

        mutator = _BatchMatmulTransposeMutator(mod)

        # Iterate through functions in the IRModule and rewrite them
        new_mod = mod  # we'll update in-place on this IRModule
        for gv, func in list(mod.functions.items()):
            if isinstance(func, relay.Function):
                new_f = mutator.visit(func)
                # Ensure the function is properly type-inferred again after mutation
                new_f = relay.Function(new_f.params, new_f.body, new_f.ret_type, new_f.type_params, new_f.attrs)
                new_mod.update_func(gv, new_f)
        # Optionally run InferType again to refresh types after insertion of transpose nodes
        new_mod = relay.transform.InferType()(new_mod)
        return new_mod
