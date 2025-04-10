def printe(message):
    import sys

    print(message, file=sys.stderr)


def _red(msg) -> str:
    return f"\x1b[31m{msg}\x1b[0m"


class TauluException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return _red(f"TauluException: {self.message}")
