class LexicaError(Exception):
    """
    User-facing, structured error.

    These errors are safe to show directly to engineers
    and LLMs without leaking stack traces.
    """

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
