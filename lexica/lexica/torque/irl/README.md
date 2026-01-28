# IRL - Lowered Intermediate Representation

## 1. Purpose of IRL

IRL, or Intermediate Representation - Lowered, is the execution-ready form of Lexica IR used internally by the Torque compiler.

Lexica IR encodes modeling intent. IRL encodes **explicit execution semantics**.

IRL exists to bridge the gap between intent-level modeling and deterministic kernel execution.

IRL is not human-written.
It is produced exclusively by the compiler.

---

## 2. Why IRL Exists

Lexica IR intentionally avoids:

* explicit body identifiers
* explicit topology references
* execution ordering details
* kernel-specific concerns

However, the CAD kernel requires all of these to be explicit.

IRL is the stage where:

* ambiguity is eliminated
* identities are assigned
* dependencies are made explicit

---

## 3. Key Properties of IRL

IRL has the following properties:

* Fully explicit
* Deterministic
* Kernel-safe
* Topology-aware
* Free of intent ambiguity

Every IRL program represents a single, fully specified execution plan.

---

## 4. IR to IRL Lowering

The lowering process converts Lexica IR into IRL.

Lowering responsibilities include:

* Assigning stable body identifiers
* Resolving implicit dependencies
* Expanding high-level operations
* Preparing kernel-ready parameters

Lowering is deterministic.
The same IR always lowers to the same IRL.

---

## 5. IRL Structure

An IRL program consists of an ordered list of IRL operations.

Each operation:

* explicitly references input bodies
* explicitly declares output bodies
* contains fully resolved parameters

There are no optional or inferred fields at this stage.

---

## 6. Validation at IRL Level

IRL undergoes a second validation phase.

IRL validation ensures:

* all body references are valid
* no undefined topology exists
* kernel preconditions are satisfied

If IRL validation fails, execution does not proceed.

---

## 7. Relationship to the Kernel

The kernel executes IRL directly.

The kernel assumes:

* IRL is structurally correct
* IRL has already been validated
* all semantics are explicit

The kernel does not perform:

* intent inference
* safety checks for malformed programs
* schema validation

---

## 8. Non-goals

IRL is not:

* a user-facing language
* a serialization format for storage
* a stable public API

IRL may evolve freely as long as IR semantics remain stable.

---

## 9. Summary

IRL is the internal execution contract of the Torque compiler.

It exists to convert intent into certainty.

Lexica IR describes what should be built.
IRL specifies exactly how it will be built.
