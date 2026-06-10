"""Async micro-batching for concurrent single-request traffic."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.serving.predictors.base import PredictionResult

if TYPE_CHECKING:
    from src.serving.predictors.batched import BatchedSklearnPredictor

logger = logging.getLogger(__name__)


@dataclass
class _BatchItem:
    text: str
    future: asyncio.Future


class AsyncBatchProcessor:
    """
    Collects concurrent `/predict` calls and flushes them as one vectorized batch.

    This implements request-level batching on top of the batched predictor,
    improving throughput under concurrent load without requiring clients to
    send explicit batch payloads.
    """

    def __init__(
        self,
        predictor: BatchedSklearnPredictor,
        max_batch_size: int = 32,
        max_wait_ms: float = 25.0,
    ):
        self._predictor = predictor
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._queue: list[_BatchItem] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def predict(self, text: str) -> PredictionResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        item = _BatchItem(text=text, future=future)

        async with self._lock:
            self._queue.append(item)
            should_flush_now = len(self._queue) >= self._max_batch_size
            if not should_flush_now and self._flush_task is None:
                self._flush_task = asyncio.create_task(self._delayed_flush())

        if should_flush_now:
            await self._flush()

        return await future

    async def _delayed_flush(self) -> None:
        try:
            await asyncio.sleep(self._max_wait_s)
            await self._flush()
        except asyncio.CancelledError:
            return

    async def _flush(self) -> None:
        async with self._lock:
            flush_task = self._flush_task
            self._flush_task = None
            # Do not cancel the task that is currently executing _flush.
            current = asyncio.current_task()
            if flush_task and flush_task is not current and not flush_task.done():
                flush_task.cancel()

            if not self._queue:
                return

            batch = self._queue
            self._queue = []

        texts = [item.text for item in batch]
        try:
            results = await asyncio.to_thread(self._predictor.predict_batch, texts)
            for item, result in zip(batch, results, strict=True):
                if not item.future.done():
                    item.future.set_result(result)
        except Exception as exc:
            logger.exception("Batch inference failed")
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)

    async def shutdown(self) -> None:
        await self._flush()
