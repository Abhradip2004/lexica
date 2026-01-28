# Torque Kernel Execution Model

## 1. Role of the Kernel

The Torque kernel is the lowest-level execution component of the Torque compiler. It is responsible for **executing IRL operations** and producing concrete geometry using a CAD backend.

The kernel does not understand Lexica IR. It only understands IRL.

The kernel is intentionally minimal and strict.

---

## 2. Responsibilities

The kernel is responsible for:

* Executing IRL operations in order
* Managing explicit body state
* Applying geometric primitives, features, transforms, and booleans
* Maintaining a deterministic execution context
* Exporting final geometry

The kernel operates purely on fully specified instructions.

---

## 3. Non-Responsibilities

The kernel explicitly does not:

* Validate Lexica IR
* Infer user intent
* Resolve ambiguous topology
* Guess missing parameters
* Perform schema validation
* Recover from malformed input

If malformed IRL reaches the kernel, behavior is undefined by design.

---

## 4. Execution Model

The kernel executes IRL operations sequentially.

Each IRL operation:

* declares its input bodies explicitly
* produces zero or more output bodies
* mutates kernel state in a deterministic way

There is no implicit state sharing or hidden execution order.

---

## 5. Kernel State

The kernel maintains an explicit execution state that includes:

* a mapping of body identifiers to geometry objects
* the active working context
* execution metadata required for export

All state transitions are explicit and reproducible.

---

## 6. Adapters

Kernel adapters translate IRL operations into CAD backend calls.

Examples:

* primitive adapters
* feature adapters
* transform adapters
* boolean adapters
* export adapters

Each adapter:

* implements a single operation category
* assumes validated input
* performs no inference

Adapters are the only components allowed to call the CAD backend directly.

---

## 7. Error Handling

Kernel errors are treated as fatal execution errors.

The kernel:

* does not attempt partial recovery
* does not continue after failure
* surfaces errors explicitly

All recoverable errors must be handled before kernel execution.

---

## 8. Determinism Guarantees

Given the same IRL input, the kernel guarantees:

* identical execution order
* identical geometry construction
* identical exported output

No randomness or environment-dependent behavior is permitted.

---

## 9. Extending the Kernel

New operations should be added by:

1. Extending IRL
2. Adding a corresponding adapter
3. Updating lowering logic

The kernel should never be extended directly to accept higher-level intent.

---

## 10. Summary

The kernel is the execution backbone of Torque.

It assumes correctness, enforces determinism, and performs no interpretation.

All intelligence lives above the kernel.
