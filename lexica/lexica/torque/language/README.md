# Lexica IR v0 Specification

## 1. Scope and Purpose

Lexica IR is the source language consumed by the Torque compiler. It encodes **modeling intent** in a strict, deterministic, and kernel-agnostic form. Lexica IR is validated statically and lowered into IRL before any geometry is executed.

This document is normative for IR v0.

---

## 2. Design Goals

* Deterministic and reproducible
* Strongly typed via explicit enums
* Intent-level, not execution-level
* Topology-agnostic at IR level
* Human-writable, machine-validated

Non-goals are listed in Section 9.

---

## 3. Program Structure

A Lexica IR program is a single JSON object with an ordered list of operations.

```json
{
  "ops": [ /* ordered operations */ ]
}
```

Execution order is the order of the list. Later operations may depend implicitly on earlier ones.

---

## 4. Operation Kinds

Each operation has a required field `kind` that selects the operation category. Additional required fields depend on the category.

Valid `kind` values in IR v0:

* `primitive`
* `feature`
* `transform`
* `boolean`
* `export`

---

## 5. Primitive Operation

Creates a base geometric primitive.

### Required Fields

* `kind`: `"primitive"`
* `primitive_kind`: enum value
* `params`: object

### Supported `primitive_kind` Values

* `box`
* `cylinder`
* `sphere`
* `cone`
* `torus`

### Parameters

Parameters are primitive-specific and validated by the language layer.

### Example

```json
{
  "kind": "primitive",
  "primitive_kind": "box",
  "params": {
    "x": 10,
    "y": 20,
    "z": 5
  }
}
```

---

## 6. Feature Operation

Applies a modeling feature to an existing body.

### Required Fields

* `kind`: `"feature"`
* `feature_kind`: enum value
* `params`: object

Topology references are not resolved at IR level and are deferred to IRL.

---

## 7. Transform Operation

Applies a geometric transformation.

### Required Fields

* `kind`: `"transform"`
* `transform_kind`: enum value
* `params`: object

---

## 8. Boolean Operation

Combines bodies using boolean logic.

### Required Fields

* `kind`: `"boolean"`
* `boolean_kind`: enum value
* `params`: object

Operand resolution is deferred to IRL.

---

## 9. Export Operation

Exports the final result of the program.

### Required Fields

* `kind`: `"export"`
* `format`: output format
* `path`: output file path

### Example

```json
{
  "kind": "export",
  "format": "step",
  "path": "output.step"
}
```

---

## 10. Validation Rules

The IR validator enforces:

* Structural correctness of all operations
* Valid enum values
* Presence of required fields
* Legal operation ordering

The validator does not:

* Resolve topology
* Execute geometry
* Infer missing parameters

---

## 11. Minimal Working Program

```json
{
  "ops": [
    {
      "kind": "primitive",
      "primitive_kind": "box",
      "params": { "x": 10, "y": 20, "z": 5 }
    },
    {
      "kind": "export",
      "format": "step",
      "path": "box.step"
    }
  ]
}
```

---

## 12. Non-goals

Lexica IR is not:

* A scripting language
* A CAD kernel API
* A topology-level language
* A heuristic or fuzzy system

All ambiguity is resolved during lowering and execution, not in IR.
