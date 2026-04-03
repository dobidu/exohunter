"""Tests for the utils module (logging, timing, parallel).

Verifies that the shared infrastructure works correctly.
"""

import time

import numpy as np
import pytest


# Top-level functions for ProcessPoolExecutor (must be picklable)
def _square(x: int) -> int:
    return x ** 2


class TestGetLogger:
    """Test the structured logging factory."""

    def test_returns_logger(self) -> None:
        """get_logger must return a standard library Logger."""
        import logging
        from exohunter.utils.logging import get_logger

        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_no_duplicate_handlers(self) -> None:
        """Calling get_logger twice must not add duplicate handlers."""
        from exohunter.utils.logging import get_logger

        logger = get_logger("test_dup")
        n_handlers_first = len(logger.handlers)

        get_logger("test_dup")
        assert len(logger.handlers) == n_handlers_first


class TestTimingDecorator:
    """Test the @timing performance measurement decorator."""

    def test_preserves_return_value(self) -> None:
        """Decorated function must return the same value as undecorated."""
        from exohunter.utils.timing import timing

        @timing
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_preserves_function_name(self) -> None:
        """Decorated function must preserve __name__ via functools.wraps."""
        from exohunter.utils.timing import timing

        @timing
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"


class TestParallelHelpers:
    """Test the thread/process pool wrappers."""

    def test_parallel_threads_preserves_order(self) -> None:
        """Results must be returned in submission order, not completion order."""
        from exohunter.utils.parallel import run_parallel_threads

        def double(x: int) -> int:
            return x * 2

        results = run_parallel_threads(double, [1, 2, 3, 4, 5], max_workers=2)
        assert results == [2, 4, 6, 8, 10]

    def test_parallel_threads_handles_exceptions(self) -> None:
        """Failed items must be None in the results list."""
        from exohunter.utils.parallel import run_parallel_threads

        def maybe_fail(x: int) -> int:
            if x == 3:
                raise ValueError("boom")
            return x * 2

        results = run_parallel_threads(maybe_fail, [1, 2, 3, 4], max_workers=2)
        assert results[0] == 2
        assert results[1] == 4
        assert results[2] is None  # failed
        assert results[3] == 8

    def test_parallel_processes_preserves_order(self) -> None:
        """Process pool must also preserve submission order."""
        from exohunter.utils.parallel import run_parallel_processes

        # Must use a top-level function (not lambda/closure) for pickling
        results = run_parallel_processes(_square, [1, 2, 3, 4], max_workers=2)
        assert results == [1, 4, 9, 16]

    def test_parallel_threads_empty_input(self) -> None:
        """Empty input must return empty list without crashing."""
        from exohunter.utils.parallel import run_parallel_threads

        results = run_parallel_threads(lambda x: x, [], max_workers=2)
        assert results == []
