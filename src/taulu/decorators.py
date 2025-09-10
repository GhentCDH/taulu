import logging
import functools
import datetime
from typing import Any, Callable, Optional

# Configure logging format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def log_calls(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    include_return: bool = False,
    max_arg_length: int = 100,
) -> Callable:
    """
    Decorator to log function calls with their arguments and timestamp.

    Args:
        logger: Logger instance to use (defaults to function's module logger)
        level: Logging level (default: INFO)
        include_return: Whether to also log return values
        max_arg_length: Maximum length for argument representation

    Usage:
        @log_calls()
        def my_function(x, y):
            return x + y

        @log_calls(level=logging.DEBUG, include_return=True)
        def another_function(data):
            return processed_data
    """

    def decorator(func: Callable) -> Callable:
        # Get logger for the function's module if none provided
        func_logger = logger or logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Format arguments
            arg_strs = []

            # Positional arguments
            for i, arg in enumerate(args):
                arg_repr = _format_arg(arg, max_arg_length)
                arg_strs.append(f"arg{i}={arg_repr}")

            # Keyword arguments
            for key, value in kwargs.items():
                value_repr = _format_arg(value, max_arg_length)
                arg_strs.append(f"{key}={value_repr}")

            args_str = ", ".join(arg_strs)

            # Log function call
            func_logger.log(
                level,
                f"CALL: {func.__name__}({args_str}) at {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}",
            )

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Log return value if requested
                if include_return:
                    result_repr = _format_arg(result, max_arg_length)
                    func_logger.log(level, f"RETURN: {func.__name__} -> {result_repr}")

                return result

            except Exception as e:
                # Log exceptions
                func_logger.log(
                    logging.ERROR,
                    f"EXCEPTION: {func.__name__} raised {type(e).__name__}: {str(e)}",
                )
                raise

        return wrapper

    return decorator


def _format_arg(arg: Any, max_length: int) -> str:
    """Format an argument for logging, truncating if too long."""
    try:
        if hasattr(arg, "shape"):  # numpy arrays, etc.
            return f"{type(arg).__name__}(shape={arg.shape}, dtype={getattr(arg, 'dtype', 'unknown')})"
        elif hasattr(arg, "__len__") and not isinstance(arg, str):
            return f"{type(arg).__name__}(len={len(arg)})"
        else:
            repr_str = repr(arg)
            if len(repr_str) > max_length:
                return repr_str[: max_length - 3] + "..."
            return repr_str
    except Exception as _:
        return f"<{type(arg).__name__} object>"
