import cadquery as cq
from lexica.cad_engine.kernel.cq_kernel import CadQueryKernel, IROp



def test_box_creation():
    kernel = CadQueryKernel()

    ir = [
        IROp(
            op="box",
            params={"x": 10, "y": 20, "z": 30}
        )
    ]

    result = kernel.execute(ir)

    assert not result.errors
    assert result.solid is not None

    bbox = result.solid.BoundingBox()
    assert bbox.xlen == 10
    assert bbox.ylen == 20
    assert bbox.zlen == 30
