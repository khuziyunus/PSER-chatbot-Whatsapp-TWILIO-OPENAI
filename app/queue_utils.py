import queue
import threading
from typing import Callable, Tuple, Any

from app.logger_utils import logger


class MessageDispatcher:
    """Simple threaded queue for non-blocking outbound processing."""

    def __init__(self, worker_fn: Callable[..., None], worker_count: int = 2) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        self._worker_fn = worker_fn
        self._queue: "queue.Queue[Tuple[tuple[Any, ...], dict[str, Any]]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        for idx in range(worker_count):
            thread = threading.Thread(target=self._worker, name=f"message-dispatcher-{idx}", daemon=True)
            thread.start()
            self._threads.append(thread)
        logger.info("MessageDispatcher started with %d worker(s)", worker_count)

    def enqueue(self, *args: Any, **kwargs: Any) -> None:
        """
        Add a message task to the queue. Args/kwargs must match the signature of worker_fn.
        """
        self._queue.put((args, kwargs))

    def _worker(self) -> None:
        """Background worker loop that pulls tasks and invokes `worker_fn`."""
        while not self._stop_event.is_set():
            try:
                args, kwargs = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._worker_fn(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - best-effort logging
                logger.exception("MessageDispatcher worker failed: %s", exc)
            finally:
                self._queue.task_done()

    def pending_tasks(self) -> int:
        """Return the approximate number of queued tasks."""
        return self._queue.qsize()

"""Threaded message dispatch queue.

Provides a small worker pool to process outbound tasks without blocking
request handlers. Workers invoke a provided `worker_fn` with queued args.
"""
