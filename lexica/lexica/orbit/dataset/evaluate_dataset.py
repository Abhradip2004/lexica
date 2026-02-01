import os
import json
from collections import defaultdict
from typing import Dict, Any, Tuple

from tqdm import tqdm

from lexica.torque.language.ir.schema import (
    IRModel,
    PrimitiveOp,
    FeatureOp,
    TransformOp,
    BooleanOp,
    ExportOp,
    PrimitiveKind,
    FeatureKind,
    TransformKind,
    BooleanKind,
    TopologyIntent,
    TopologyTarget,
)


from lexica.torque.language.ir.validate import validate_ir
from lexica.torque.irl.ir_to_irl import lower_ir_to_irl
from lexica.torque.irl.validation import validate_irl
from lexica.torque.kernel.executor import IRLExecutor


# -----------------------------
# Config
# -----------------------------

RAW_DATASET = "raw_dataset.jsonl"
FINAL_DATASET = "final_dataset.jsonl"
FAILED_DATASET = "failed_dataset.jsonl"
REPORT_FILE = "eval_report.json"

SAVE_STEP_PER_SAMPLE = True
STEP_OUT_DIR = "step_outputs"


# -----------------------------
# Helpers
# -----------------------------

def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def parse_topology_intent(d: Dict[str, Any]) -> TopologyIntent:
    """
    Convert dict -> TopologyIntent dataclass.
    Example:
      {"target":"edge", "rule":"all"}
    """
    if d is None:
        return None

    target = d.get("target")
    rule = d.get("rule")
    value = d.get("value", None)

    # Convert target string -> enum
    if isinstance(target, str):
        target = TopologyTarget(target)

    return TopologyIntent(target=target, rule=rule, value=value)


def parse_op(op_dict: Dict[str, Any]):
    """
    Convert a single op dict back into the correct Op dataclass.
    """
    kind = op_dict.get("kind")
    if not kind:
        raise ValueError("Op dict missing 'kind'")

    # ---- Primitive ----
    if kind == "primitive":
        pk = op_dict.get("primitive_kind")
        params = op_dict.get("params", {})

        if isinstance(pk, str):
            pk = PrimitiveKind(pk)

        return PrimitiveOp(primitive_kind=pk, params=params)

    # ---- Feature ----
    if kind == "feature":
        fk = op_dict.get("feature_kind")
        params = op_dict.get("params", {})
        topo = op_dict.get("topology", None)

        if isinstance(fk, str):
            fk = FeatureKind(fk)

        topology = parse_topology_intent(topo) if topo else None

        return FeatureOp(feature_kind=fk, params=params, topology=topology)

    # ---- Transform ----
    if kind == "transform":
        tk = op_dict.get("transform_kind")
        params = op_dict.get("params", {})

        if isinstance(tk, str):
            tk = TransformKind(tk)

        return TransformOp(transform_kind=tk, params=params)

    # ---- Boolean ----
    if kind == "boolean":
        bk = op_dict.get("boolean_kind")
        params = op_dict.get("params", {})

        if isinstance(bk, str):
            bk = BooleanKind(bk)

        return BooleanOp(boolean_kind=bk, params=params)

    # ---- Export ----
    if kind == "export":
        fmt = op_dict.get("format", "step")
        params = op_dict.get("params", {})
        return ExportOp(format=fmt, params=params)

    raise ValueError(f"Unknown op kind: {kind}")


def dict_to_ir_model(ir_dict: Dict[str, Any]) -> IRModel:
    """
    Convert IR dict (from json dataset) back into IRModel dataclass.
    """
    ops_raw = ir_dict.get("ops", [])
    if not isinstance(ops_raw, list):
        raise ValueError("IR dict 'ops' must be a list")

    ops = [parse_op(o) for o in ops_raw]
    return IRModel(ops=ops)


# -----------------------------
# Kernel runner
# -----------------------------

def run_ir(ir_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Run a single IR dict through:
      dict -> IRModel dataclass -> validate_ir -> lower -> validate_irl -> execute
    Returns (ok, error_message).
    """
    try:
        ir = dict_to_ir_model(ir_dict)

        validate_ir(ir)

        irl = lower_ir_to_irl(ir)
        validate_irl(irl)

        ex = IRLExecutor()
        ex.execute(irl)

        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_dataset(
    in_file: str = RAW_DATASET,
    out_file: str = FINAL_DATASET,
    failed_file: str = FAILED_DATASET,
    report_file: str = REPORT_FILE,
):
    if SAVE_STEP_PER_SAMPLE:
        os.makedirs(STEP_OUT_DIR, exist_ok=True)

    total = 0
    ok_count = 0

    ok_by_template = defaultdict(int)
    fail_by_template = defaultdict(int)
    error_examples = defaultdict(list)

    with open(in_file, "r") as fin, \
         open(out_file, "w") as fout_ok, \
         open(failed_file, "w") as fout_fail:

        for idx, line in enumerate(tqdm(fin, desc="Evaluating IR samples"), start=1):
            total += 1
            sample = json.loads(line)

            template_id = sample.get("template_id", "unknown")
            ir_dict = sample.get("ir")

            if not ir_dict:
                fail_by_template[template_id] += 1
                if len(error_examples[template_id]) < 5:
                    error_examples[template_id].append("Missing 'ir' field")
                sample["error"] = "Missing 'ir' field"
                fout_fail.write(json.dumps(sample) + "\n")
                continue

            ok, err = run_ir(ir_dict)

            if ok:
                ok_count += 1
                ok_by_template[template_id] += 1
                fout_ok.write(json.dumps(sample) + "\n")

                # Save step uniquely
                if SAVE_STEP_PER_SAMPLE:
                    src_step = "lexica_output.step"
                    if os.path.exists(src_step):
                        file_id = f"{idx:06d}_{safe_filename(template_id)}"
                        dst_step = os.path.join(STEP_OUT_DIR, f"{file_id}.step")
                        try:
                            with open(src_step, "rb") as fsrc, open(dst_step, "wb") as fdst:
                                fdst.write(fsrc.read())
                        except Exception:
                            pass
            else:
                fail_by_template[template_id] += 1
                if len(error_examples[template_id]) < 5:
                    error_examples[template_id].append(err)
                sample["error"] = err
                fout_fail.write(json.dumps(sample) + "\n")

    report = {
        "input_file": in_file,
        "output_file": out_file,
        "failed_file": failed_file,
        "total_samples": total,
        "valid_samples": ok_count,
        "invalid_samples": total - ok_count,
        "valid_ratio": (ok_count / total) if total else 0.0,
        "valid_by_template": dict(ok_by_template),
        "fail_by_template": dict(fail_by_template),
        "error_examples": dict(error_examples),
        "save_step_per_sample": SAVE_STEP_PER_SAMPLE,
        "step_output_dir": STEP_OUT_DIR if SAVE_STEP_PER_SAMPLE else None,
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\nEvaluation complete")
    print(f"Total:  {total}")
    print(f"Valid:  {ok_count}")
    print(f"Invalid:{total - ok_count}")
    print(f"Saved curated dataset --> {out_file}")
    print(f"Saved failures --> {failed_file}")
    print(f"Saved report --> {report_file}")
    if SAVE_STEP_PER_SAMPLE:
        print(f"Saved STEP outputs --> {STEP_OUT_DIR}/")


if __name__ == "__main__":
    evaluate_dataset()
