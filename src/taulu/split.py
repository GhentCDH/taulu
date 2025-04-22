from typing import Generic, TypeVar, Callable, Any

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class Split(Generic[T]):
    """Wrapper for data that has both a left and a right variant"""

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

    def __repr__(self) -> str:
        return f"left: {self._left}, right: {self._right}"

    def __iter__(self):
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

    def apply(
        self,
        funcs: "Split[Callable[[T, *Any], V]] | Callable[[T, *Any], V]",
        *args,
        **kwargs,
    ) -> "Split[V]":
        if not isinstance(funcs, Split):
            funcs = Split(funcs, funcs)

        def get_arg(side: str, arg):
            if isinstance(arg, Split):
                return getattr(arg, side)
            return arg

        def call(side: str):
            func = getattr(funcs, side)
            target = getattr(self, side)

            side_args = [get_arg(side, arg) for arg in args]
            side_kwargs = {k: get_arg(side, v) for k, v in kwargs.items()}

            return func(target, *side_args, **side_kwargs)

        return Split(call("left"), call("right"))

    def __getattr__(self, attr_name: str):
        if attr_name in self.__dict__:
            return getattr(self, attr_name)

        def wrapper(*args, **kwargs):
            return self.apply(
                Split(
                    getattr(self.left.__class__, attr_name),
                    getattr(self.right.__class__, attr_name),
                ),
                *args,
                **kwargs,
            )

        return wrapper
