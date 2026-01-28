# Torque -- IR --> CAD Compiler

## 1. What is Torque?

Torque is a deterministic compiler that converts Lexica IR, a high-level, intent-based modeling language, into concrete CAD artifacts such as STEP files.

Torque is based on and inspired by the json2CAD generator created by Abhradip Dey. It can be seen as an evolution and formalization of those ideas into a strict compiler architecture with a well-defined intermediate language and execution pipeline.

Torque is not an LLM system, not a CAD UI, and not a scripting layer. It is a strict, predictable compilation engine.

In short:

> Torque takes structured intent and produces geometry.

---

## 2. Design Philosophy

Torque is built around a few non‑negotiable principles:

### 2.1 Determinism

Given the same input IR, Torque will always produce the same output. There is no randomness, heuristics, or inference.

### 2.2 Strict Boundaries

Torque does not:

* interpret natural language
* guess missing parameters
* fix malformed input
* perform fuzzy matching

If input is invalid, Torque fails early and explicitly.

### 2.3 Compiler‑like Architecture

Torque is structured like a traditional compiler:

```
Lexica IR  -->  IR Validation  -->  IRL Lowering  -->  IRL Validation  -->  Kernel Execution  -->  CAD Output
```

Each stage has a single responsibility and clear invariants.

---

## 3. Torque Is Standalone

Torque can run entirely on its own.

It does **not** depend on:

* Orbit (LLM frontend)
* APIs
* Web services
* UI layers

You can hand‑write Lexica IR and feed it directly to Torque.

This property is intentional.

---

## 4. Core Components

### 4.1 Language Layer (`torque/language`)

Defines **Lexica IR**, the source language consumed by Torque.

Responsibilities:

* IR schema definitions
* Enum definitions
* Language‑level validation

This layer answers:

> “Is this IR a valid program?”

It does **not**:

* execute geometry
* assign topology
* reference CAD kernels

---

### 4.2 IRL Layer (`torque/irl`)

IRL (Intermediate Representation - Lowered) is the **execution‑ready** form of IR.

Responsibilities:

* Explicit body identities
* Explicit read/write semantics
* Topology resolution contracts
* Kernel‑safe validation

IRL is not human‑written.
It is produced by the compiler.

---

### 4.3 Kernel (`torque/kernel`)

The kernel is responsible for **actual geometry execution**.

Responsibilities:

* Execute IRL operations
* Manage CAD state explicitly
* Resolve topology deterministically
* Export geometry

The kernel assumes:

* IRL has already been validated
* No malformed input exists

---

### 4.4 Pipeline (`torque/pipeline`)

The pipeline connects the stages together.

Responsibilities:

* IR --> IRL lowering
* IRL execution orchestration

The pipeline contains **no policy** and **no heuristics**.

---

## 5. Torque CLI

Torque provides a minimal CLI interface for standalone usage.

Example:

```bash
python -m lexica.torque.cli examples/box_ir.json
```

The CLI:

* parses IR
* validates IR
* lowers IR --> IRL
* executes IRL

The CLI owns:

* file I/O
* argument parsing
* output paths

The kernel does not.

---

## 6. What Torque Guarantees

If Torque accepts an IR program:

* The program is structurally valid
* Execution is deterministic
* Geometry will be produced or an explicit error raised

Torque never silently recovers from errors.

---

## 7. What Torque Refuses to Do

Torque intentionally refuses to:

* infer user intent
* guess missing parameters
* correct malformed IR
* execute partially valid programs
* depend on LLM output correctness

These responsibilities belong **outside** the compiler.

---

## 8. Relationship to Orbit

Orbit is a **frontend** that generates Lexica IR from natural language.

Torque:

* does not know Orbit exists
* does not trust Orbit
* validates Orbit output like any other IR

This separation is deliberate and enforced.

---

## 9. Versioning Philosophy

Lexica IR is versioned explicitly.

Torque guarantees:

* backward compatibility within a major IR version
* explicit breakage between major versions

IR versioning is handled at the language layer, not the kernel.

---

## 10. Summary

Torque is a compiler, not a convenience layer.

It exists to provide:

* correctness
* predictability
* reproducibility

All flexibility lives outside the compiler.
