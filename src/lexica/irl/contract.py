from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Protocol


# ------------------
# IR
#-------------------

@dataclass
class IRPrimitive:
    type: str                    # Shape: box, cylinder, sphere, etc.
    params: Dict[str, Any]       # Dimensions: width, height, radius, etc.
    transforms: List[Dict[str, Any]]


@dataclass
class IRModel:
    primitives: List[IRPrimitive]
    metadata: Dict[str, Any]

#----------------------
# IRL 
#----------------------

@dataclass
class IRLOperation:
    op: str                      # create_box, translate, fillet, etc.
    args: Dict[str, Any]


@dataclass
class IRLProgram:
    operations: List[IRLOperation]
    backend: str                 # cadquery, occ, future backends

#---------------------------
# Compiler Contract 
#---------------------------

class IRToIRLCompiler(Protocol):
    def compile(self, ir: IRModel) -> IRLProgram:
        ...
