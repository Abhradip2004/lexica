from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class IR:
    entity: str
    primitive: str
    parameters: Dict[str, float]
    units: str
    constraints: List[Any]
    metadata: Dict[str, Any]
