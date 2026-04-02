"""Performance measurement decorators for ExoHunter.

The ``@timing`` decorator logs the wall-clock time of any function call,
making it easy to spot bottlenecks during pipeline runs.

Usage::

    from exohunter.utils.timing import timing

    @timing
    def expensive_computation(data):
        ...
"""

import functools
import time
from typing import Any, Callable, TypeVar

from exohunter.utils.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timing(func: F) -> F:
    """Decorator that logs the elapsed time of a function call.

    Args:
        func: The function to wrap.

    Returns:
        The wrapped function with identical signature and return value.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        # Format nicely depending on magnitude
        if elapsed < 1.0:
            time_str = f"{elapsed * 1000:.1f} ms"
        elif elapsed < 60.0:
            time_str = f"{elapsed:.2f} s"
        else:
            minutes, seconds = divmod(elapsed, 60)
            time_str = f"{int(minutes)}m {seconds:.1f}s"

        logger.info("%s completed in %s", func.__qualname__, time_str)
        return result

    return wrapper  # type: ignore[return-value]
