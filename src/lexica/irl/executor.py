from __future__ import annotations

from lexica.irl.contract import IRLProgram
from lexica.irl.cadquery_backend import CadQueryBackend


class IRLExecutor:
    """
    Executes an IRLProgram using the specified backend.
    """

    def __init__(self):
        self._backends = {
            "cadquery": CadQueryBackend,
        }

    def run(self, program: IRLProgram):
        backend_name = program.backend

        if backend_name not in self._backends:
            raise ValueError(f"Unsupported IRL backend: {backend_name}")

        backend = self._backends[backend_name]()
        return backend.execute(program)
