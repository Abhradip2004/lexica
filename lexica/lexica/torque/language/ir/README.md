# Lexica IR (v0)

Lexica IR is the source language for the Torque compiler.

It describes **modeling intent**, not execution steps.

## Properties
- Sequential
- Deterministic
- Human-writable
- Kernel-agnostic
- Strictly validated

## Usage
You can:
- Hand-write IR as JSON
- Generate IR via Orbit (LLM frontend)
- Feed IR directly into Torque

Torque will:
IR --> IRL --> CAD --> STEP

## Important
- IR is validated before lowering
- IR is never executed directly
- All execution happens at the IRL level
