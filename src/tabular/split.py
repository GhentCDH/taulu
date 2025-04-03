from typing import Generic, TypeVar

T = TypeVar("T")

class Split(Generic[T]):
    """Contains data for left and right images / processors / separately"""

    def __init__(self, left: T | None = None, right: T | None = None):
        self._left = left
        self._right = right

    @property
    def left(self) -> T:
        assert self._left is not None
        return self._left

    @left.setter
    def left(self, value: T):
        self._left = value

    @property
    def right(self) -> T:
        assert self._right is not None
        return self._right

    @right.setter
    def right(self, value: T):
        self._right = value

    def append(self, value: T):
        if self._left is None:
            self._left = value
        else:
            self._right = value

    def __iter__(self):
        """Allows unpacking with `left, right = split`"""
        assert self._left is not None
        assert self._right is not None
        return iter((self._left, self._right))

    def __getitem__(self, index: bool) -> T:
        assert self._left is not None
        assert self._right is not None
        if int(index) == 0:
            return self._left
        else:
            return self._right

