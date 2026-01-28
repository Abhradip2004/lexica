# IRL (Lowered Intermediate Representation)

IRL is the kernel-facing representation used by Torque.

Unlike Lexica IR, IRL:
- Has explicit body identities
- Has explicit read/write semantics
- Encodes body lifetimes
- Is safe to execute deterministically

## Pipeline
Lexica IR → IRL → Kernel → STEP

## Guarantees
If an IRL program passes validation:
- The kernel may execute it
- No implicit mutation occurs
- Body lifetimes are respected

IRL is not human-authored.
It is produced by the compiler.
