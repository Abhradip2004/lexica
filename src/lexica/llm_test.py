from lexica.pipeline.nl_to_ir.translator import nl_to_ir

tests = [
    "Make a box with dimensions, 30, 30, 30",
    "Make a cylinder of diameter: 10 and height: 30",
]

for t in tests:
    print("\nPROMPT:", t)
    ir = nl_to_ir(t, debug=True)
    print(ir)
