"""
Torque Kernel

This module executes validated IRL programs.

Responsibilities:
- Deterministic execution
- Explicit body lifecycle
- No implicit CAD state
- No language or LLM logic

If IRL validation passes, kernel execution is guaranteed
to be safe and deterministic.
"""

from .executor import IRLExecutor, ExecutorError
