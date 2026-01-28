# Torque Kernel

The Torque kernel executes **IRL (Lowered IR)** programs
and produces CAD artifacts.

## Properties
- Deterministic
- Explicit state
- No heuristics
- No LLM involvement
- No implicit geometry context

## Execution Model
IRL → Kernel → CAD → STEP

## Important
- The kernel never sees Lexica IR
- The kernel never performs validation beyond IRL rules
- All topology resolution is deterministic
