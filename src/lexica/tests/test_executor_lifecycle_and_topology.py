import pytest

from lexica.irl.contract import (
    IRLModel,
    PrimitiveOp,
    FeatureOp,
    BooleanOp,
    ExportOp,
    TopoPredicate,
    TopoTarget,
)

from lexica.cad_engine.executor import IRLExecutor, ExecutorError


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def run_model(ops):
    model = IRLModel(ops=ops)
    executor = IRLExecutor()
    executor.execute(model)
    return executor


# --------------------------------------------------
# Tests
# --------------------------------------------------

def test_feature_overwrite_kills_source_body():
    """
    Feature overwrite=True should replace the input body.
    """

    ops = [
        PrimitiveOp(
            op_id="box",
            reads=[],
            writes="body_0",
            params={"kind": "box", "length": 10, "width": 10, "height": 10},
        ),
        FeatureOp(
            op_id="fillet",
            reads=["body_0"],
            writes="body_1",
            overwrite=True,
            params={"kind": "fillet", "radius": 1.0},
            topo=TopoPredicate(target=TopoTarget.EDGE, rule="all"),
        ),
    ]

    executor = run_model(ops)

    # body_0 must be DEAD
    assert "body_0" not in executor.registry._bodies
    assert "body_1" in executor.registry._bodies


def test_feature_fork_preserves_source_body():
    """
    Feature overwrite=False should fork and preserve source body.
    """

    ops = [
        PrimitiveOp(
            op_id="box",
            reads=[],
            writes="body_0",
            params={"kind": "box", "length": 10, "width": 10, "height": 10},
        ),
        FeatureOp(
            op_id="fillet_fork",
            reads=["body_0"],
            writes="body_1",
            overwrite=False,
            params={"kind": "fillet", "radius": 1.0},
            topo=TopoPredicate(target=TopoTarget.EDGE, rule="all"),
        ),
    ]

    executor = run_model(ops)

    assert "body_0" in executor.registry._bodies
    assert "body_1" in executor.registry._bodies


def test_boolean_consumes_all_input_bodies():
    """
    Boolean ops must consume all input bodies.
    """

    ops = [
        PrimitiveOp(
            op_id="box",
            reads=[],
            writes="a",
            params={"kind": "box", "length": 10, "width": 10, "height": 10},
        ),
        PrimitiveOp(
            op_id="cyl",
            reads=[],
            writes="b",
            params={"kind": "cylinder", "radius": 2, "height": 10},
        ),
        BooleanOp(
            op_id="cut",
            reads=["a", "b"],
            writes="c",
            kind="difference",
        ),
    ]

    executor = run_model(ops)

    assert "a" not in executor.registry._bodies
    assert "b" not in executor.registry._bodies
    assert "c" in executor.registry._bodies


def test_topology_zero_selection_fails():
    """
    Topology rules that match nothing must fail loudly.
    """

    ops = [
        PrimitiveOp(
            op_id="box",
            reads=[],
            writes="body_0",
            params={"kind": "box", "length": 10, "width": 10, "height": 10},
        ),
        FeatureOp(
            op_id="bad_fillet",
            reads=["body_0"],
            writes="body_1",
            overwrite=True,
            params={"kind": "fillet", "radius": 1.0},
            topo=TopoPredicate(
                target=TopoTarget.EDGE,
                rule="by_length_gt",
                value=1e9,  # impossible
            ),
        ),
    ]

    with pytest.raises(Exception):
        run_model(ops)


def test_missing_topology_for_fillet_fails():
    """
    Fillet without topology must fail before kernel execution.
    """

    ops = [
        PrimitiveOp(
            op_id="box",
            reads=[],
            writes="body_0",
            params={"kind": "box", "length": 10, "width": 10, "height": 10},
        ),
        FeatureOp(
            op_id="fillet_no_topo",
            reads=["body_0"],
            writes="body_1",
            overwrite=True,
            params={"kind": "fillet", "radius": 1.0},
            topo=None,
        ),
    ]

    with pytest.raises(ExecutorError):
        run_model(ops)
