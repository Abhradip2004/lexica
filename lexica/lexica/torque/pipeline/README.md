# Torque Compiler Pipeline

## 1. Purpose of the Pipeline

The pipeline connects the major stages of the Torque compiler into a single, deterministic execution flow.

It exists to **orchestrate** stages, not to introduce logic.

The pipeline is intentionally thin.

---

## 2. Pipeline Overview

The Torque compilation pipeline follows a fixed sequence:

```
Lexica IR
  --> IR Validation
  --> IR to IRL Lowering
  --> IRL Validation
  --> Kernel Execution
  --> CAD Output
```

Each stage must complete successfully before the next stage begins.

---

## 3. Stage Responsibilities

### 3.1 IR Validation

Validates Lexica IR against the language specification.

Responsibilities:

* Structural validation
* Enum validation
* Required field checks
* Operation ordering rules

Failure at this stage stops the pipeline.

---

### 3.2 IR to IRL Lowering

Converts intent-level IR into execution-ready IRL.

Responsibilities:

* Assigning explicit body identifiers
* Resolving implicit dependencies
* Eliminating ambiguity
* Producing kernel-safe instructions

Lowering is deterministic and repeatable.

---

### 3.3 IRL Validation

Validates IRL before execution.

Responsibilities:

* Verifying body references
* Ensuring topology is fully resolved
* Checking kernel preconditions

IRL validation exists to protect the kernel.

---

### 3.4 Kernel Execution

Executes IRL operations using the kernel.

Responsibilities:

* Applying operations sequentially
* Managing kernel state
* Exporting geometry

The kernel assumes all validation has already occurred.

---

## 4. Error Handling

Errors are handled at the stage where they occur.

The pipeline:

* does not attempt recovery
* does not skip stages
* does not continue after failure

Errors are surfaced explicitly to the caller.

---

## 5. Determinism

The pipeline guarantees determinism by enforcing:

* fixed stage ordering
* explicit data flow between stages
* no hidden state

The same input IR always results in the same output.

---

## 6. What the Pipeline Does Not Do

The pipeline does not:

* interpret intent
* modify IR semantics
* perform validation outside its stage
* contain execution heuristics

All logic belongs inside individual stages.

---

## 7. Relationship to the CLI

The CLI invokes the pipeline.

The CLI owns:

* argument parsing
* file I/O
* user-facing errors

The pipeline owns:

* execution order
* stage orchestration

---

## 8. Summary

The pipeline is the backbone of Torque execution.

It connects strict stages without adding intelligence.

All correctness is enforced by structure, not heuristics.
