"""Compatibility shim for the renamed CP scheduler module.

Use `match.opt.cp` directly.
"""

from match.opt.cp import ConstraintProgrammingEngine

__all__ = ["ConstraintProgrammingEngine"]
