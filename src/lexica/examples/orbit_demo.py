from lexica.pipeline.nl_to_ir.translator import nl_to_ir

tests = [
    "Create a box of size 10 20 30",
    # "Make a cube of side 15",
    "Create a box of size 30 height, 20 width, 60 depth. Chamfer all edges by 2",
    # "Create a cylinder of diameter: 10 and height 20. Fillet all edges with radius 1",
    "Create 30 by 40 by 60 box and drill a hole of diameter 6 through the center",
    # "Move the object by 10 units along x",
    # "Rotate the object 90 degrees around z axis",
]

for t in tests:
    print("\nPROMPT:", t)
    ir = nl_to_ir(t)
    print(ir)
