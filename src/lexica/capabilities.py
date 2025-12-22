from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Capability:
    name: str
    supported: bool
    notes: str = ""


def get_capabilities() -> Dict[str, List[Capability]]:
    """
    Canonical declaration of what Lexica supports TODAY.

    This is intentionally explicit and conservative.
    """
    return {
        "primitives": [
            Capability("box", True),
            Capability("cylinder", True),
        ],
        "features": [
            Capability("fillet", True, "Edge-based fillet"),
            Capability("chamfer", True, "Edge-based chamfer"),
            Capability("shell", True, "Uniform inward shell"),
            Capability(
                "hole",
                True,
                "Through / blind hole with optional counterbore or countersink",
            ),
            Capability(
                "threads",
                False,
                "Planned (cosmetic threads first)",
            ),
        ],
        "booleans": [
            Capability("union", True),
            Capability("cut", True),
            Capability("intersect", True),
        ],
        "export": [
            Capability("STEP", True),
        ],
    }
