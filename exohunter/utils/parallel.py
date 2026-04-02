"""Parallelism helpers for ExoHunter.

Provides thin wrappers around ``concurrent.futures`` executors so that
every module uses consistent pool sizes and progress reporting.

Design decisions:
    - **ThreadPoolExecutor** for I/O-bound work (network downloads).
    - **ProcessPoolExecutor** for CPU-bound work (preprocessing, BLS).
    - ``tqdm`` progress bars are used when running interactively.

Usage::

    from exohunter.utils.parallel import run_parallel_threads

    results = run_parallel_threads(download_one, urls, max_workers=8)
"""

import os
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Any, Callable, Iterable, TypeVar

from tqdm import tqdm

from exohunter.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def _default_cpu_workers() -> int:
    """Return a sensible default number of workers for CPU-bound tasks.

    Uses ``os.cpu_count() - 1`` so that one core remains available for
    the main process and OS housekeeping.
    """
    cpu_count = os.cpu_count() or 2
    return max(1, cpu_count - 1)


def run_parallel_threads(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: int = 8,
    description: str = "Downloading",
) -> list[R]:
    """Execute *func* on each item using a thread pool.

    Suited for I/O-bound tasks such as network downloads.

    Args:
        func: A callable that takes a single item and returns a result.
        items: An iterable of inputs to process.
        max_workers: Maximum number of concurrent threads.
        description: Label shown on the progress bar.

    Returns:
        A list of results in the order they were *submitted* (not completed).
        Items whose processing raised an exception are represented as ``None``
        in the output list, and the exception is logged.
    """
    items_list = list(items)
    results: list[R | None] = [None] * len(items_list)

    logger.info(
        "Starting %s with %d items on %d threads",
        description,
        len(items_list),
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map each Future back to its original index
        future_to_index: dict[Future[R], int] = {}
        for index, item in enumerate(items_list):
            future = executor.submit(func, item)
            future_to_index[future] = index

        with tqdm(total=len(items_list), desc=description, unit="item") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception(
                        "Thread task failed for item %s", items_list[idx]
                    )
                pbar.update(1)

    # Filter out None entries (failed items)
    successful = [r for r in results if r is not None]
    logger.info(
        "%s finished: %d/%d succeeded",
        description,
        len(successful),
        len(items_list),
    )
    return results  # type: ignore[return-value]


def run_parallel_processes(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: int | None = None,
    description: str = "Processing",
) -> list[R]:
    """Execute *func* on each item using a process pool.

    Suited for CPU-bound tasks such as preprocessing or BLS computation.

    Args:
        func: A callable that takes a single item and returns a result.
            Must be picklable (top-level function, not a lambda or closure).
        items: An iterable of inputs to process.
        max_workers: Number of worker processes. Defaults to ``cpu_count - 1``.
        description: Label shown on the progress bar.

    Returns:
        A list of results in submission order. Failed items are ``None``.
    """
    if max_workers is None:
        max_workers = _default_cpu_workers()

    items_list = list(items)
    results: list[R | None] = [None] * len(items_list)

    logger.info(
        "Starting %s with %d items on %d processes",
        description,
        len(items_list),
        max_workers,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index: dict[Future[R], int] = {}
        for index, item in enumerate(items_list):
            future = executor.submit(func, item)
            future_to_index[future] = index

        with tqdm(total=len(items_list), desc=description, unit="item") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception(
                        "Process task failed for item %s", items_list[idx]
                    )
                pbar.update(1)

    successful = [r for r in results if r is not None]
    logger.info(
        "%s finished: %d/%d succeeded",
        description,
        len(successful),
        len(items_list),
    )
    return results  # type: ignore[return-value]
