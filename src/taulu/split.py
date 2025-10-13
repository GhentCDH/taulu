"""
A module that provides a Split class to handle data with left and right variants.

The Split class allows for easy management and manipulation of paired data, such as images or templates, by providing properties and methods to access and modify the left and right components. It also supports applying functions to both components simultaneously and accessing attributes of the contained objects.
"""

from typing import Generic, TypeVar, Callable, Any

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class Split(Generic[T]):
    """
    Container for paired left/right data with convenient manipulation methods.

    The Split class is designed for working with table images that span two pages
    or have distinct left and right sections. It allows you to:
    - Store related data for both sides
    - Apply functions to both sides simultaneously
    - Access attributes/methods of contained objects transparently

    Examples:
        >>> # Create a split with different parameters for each side
        >>> thresholds = Split(0.25, 0.30)
        >>>
        >>> # Apply a function to both sides
        >>> images = Split(left_img, right_img)
        >>> processed = images.apply(lambda img: cv2.blur(img, (5, 5)))
        >>>
        >>> # Use with different parameters per side
        >>> results = images.apply(
        ...     lambda img, k: sauvola_threshold(img, k),
        ...     k=thresholds  # k.left used for left img, k.right for right
        ... )
        >>>
        >>> # Access methods of contained objects directly
        >>> templates = Split(template_left, template_right)
        >>> widths = templates.cell_widths(0)  # Calls on both templates

    Type Parameters:
        T: The type of objects stored in left and right
    """

    def __init__(self, left: T | None = None, right: T | None = None):
        """
        Initialize a Split container.

        Args:
            left: Data for the left side
            right: Data for the right side

        Note:
            Both can initially be None. Use the `append` method or set
            properties directly to populate.
        """
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
