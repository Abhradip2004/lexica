from lexica.pipeline.nl_to_ir.translator import _dict_to_ir

_dict_to_ir({
  "ops": [
    {
      "kind": "primitive",
      "primitive_kind": "box",
      "params": {"length": 10, "width": 10, "height": 10}
    }
  ]
})
