"""Bounded, ordered local work with durable failure recovery state."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import CancelledError, Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, SimpleQueue

from summarizer.cache import CacheDescriptor, CacheStore
from summarizer.checkpoint import (
    CheckpointSession,
    CompletedRef,
    NonReusableReason,
    NonReusableRef,
)
from summarizer.runtime.observers import PipelineStopped

# How often a scheduler waiting on in-flight work checks for a stop request.
# Only a scheduler given a stop check waits in slices; others block outright.
STOP_POLL_SECONDS = 0.2


@dataclass(frozen=True)
class ScheduledWork:
    """One independently executable, cacheable unit with a stable work ID."""

    descriptor: CacheDescriptor
    operation: Callable[[], object]
    validate: Callable[[object], object]

    @property
    def work_id(self) -> str:
        return self.descriptor.work_id


@dataclass(frozen=True)
class ScheduledResult:
    work_id: str
    payload: object


class BoundedScheduler:
    """Run independent work locally without changing result or level order.

    Every item is recorded in the checkpoint manifest the moment it finishes,
    so a kill at any point loses only work that was still in flight, and a
    resumed run reuses everything that finished. `on_complete` then receives
    the durable result on the calling thread.

    `should_stop` is polled before each submission and after each completion,
    and at least every `STOP_POLL_SECONDS` while waiting. Once it returns true
    the scheduler submits nothing more, cancels work that has not started,
    keeps what already finished, and raises `PipelineStopped` without waiting
    for in-flight calls; their results are discarded and recomputed on resume.
    """

    def __init__(
        self,
        *,
        max_in_flight: int,
        cache: CacheStore,
        executor_factory: Callable[[int], Executor] = ThreadPoolExecutor,
        should_stop: Callable[[], bool] | None = None,
        on_complete: Callable[[ScheduledResult], None] | None = None,
    ) -> None:
        if not isinstance(max_in_flight, int) or isinstance(max_in_flight, bool):
            raise TypeError("max_in_flight must be an integer")
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        self._max_in_flight = max_in_flight
        self._cache = cache
        self._executor_factory = executor_factory
        self._should_stop = should_stop
        self._on_complete = on_complete

    def run(
        self,
        work: tuple[ScheduledWork, ...],
        session: CheckpointSession,
    ) -> tuple[ScheduledResult, ...]:
        self._validate_work(work, session)
        if not work:
            return ()

        positions = {
            work_id: index for index, work_id in enumerate(session.manifest.work_ids)
        }
        results: dict[str, ScheduledResult] = {}
        non_reusable: dict[str, NonReusableReason] = {}
        pending: dict[Future[object], ScheduledWork] = {}
        completions: SimpleQueue[Future[object]] = SimpleQueue()
        next_index = 0
        terminal_error: Exception | None = None
        terminal_work_id: str | None = None
        cleanup_error: Exception | None = None
        stopped = False
        executor: Executor | None = None

        def record_non_reusable(work_id: str, reason: NonReusableReason) -> None:
            non_reusable.setdefault(work_id, reason)

        def collect(future: Future[object], item: ScheduledWork) -> Exception | None:
            nonlocal stopped
            try:
                payload = future.result()
            except CancelledError as error:
                record_non_reusable(item.work_id, NonReusableReason.CANCELLED)
                return error
            except PipelineStopped:
                # The operation saw the stop request before its next call.
                record_non_reusable(item.work_id, NonReusableReason.CANCELLED)
                stopped = True
                return None
            except Exception as error:  # noqa: BLE001 - checkpoint all worker failures
                record_non_reusable(item.work_id, NonReusableReason.FAILED)
                return error
            try:
                validated = self._cache.store_winner(
                    item.descriptor, payload, item.validate
                )
                self._cache.record_descriptor_projection(item.descriptor)
                session.checkpoint_scheduler_state(
                    completed=(
                        CompletedRef(
                            work_id=item.work_id,
                            cache_key=item.descriptor.key,
                        ),
                    ),
                    descriptors={item.work_id: item.descriptor},
                )
            except Exception as error:  # noqa: BLE001 - preserve validation failures
                record_non_reusable(item.work_id, NonReusableReason.FAILED)
                return error
            result = ScheduledResult(item.work_id, validated)
            results[item.work_id] = result
            if self._on_complete is not None:
                self._on_complete(result)
            return None

        def collect_finished(finished: Iterable[Future[object]]) -> None:
            nonlocal terminal_error, terminal_work_id
            for future in finished:
                item = pending.pop(future)
                error = collect(future, item)
                if terminal_error is None and error is not None:
                    terminal_error = error
                    terminal_work_id = item.work_id

        def in_plan_order(futures: Iterable[Future[object]]) -> list[Future[object]]:
            return sorted(futures, key=lambda future: positions[pending[future].work_id])

        try:
            try:
                executor = self._executor_factory(self._max_in_flight)
            except Exception as error:  # noqa: BLE001 - no executor was observable
                terminal_error = error

            while (
                executor is not None
                and terminal_error is None
                and not stopped
                and (pending or next_index < len(work))
            ):
                while next_index < len(work) and len(pending) < self._max_in_flight:
                    if self._stop_requested():
                        stopped = True
                        break
                    item = work[next_index]
                    next_index += 1
                    try:
                        future = executor.submit(item.operation)
                    except Exception as error:  # noqa: BLE001 - future is unavailable
                        record_non_reusable(
                            item.work_id, NonReusableReason.UNOBSERVABLE
                        )
                        terminal_error = error
                        terminal_work_id = item.work_id
                        break
                    pending[future] = item
                    future.add_done_callback(completions.put)
                if stopped or terminal_error is not None:
                    break
                collect_finished(in_plan_order(self._finished(completions, pending)))
                if self._stop_requested():
                    stopped = True

            if terminal_error is not None and not stopped:
                # Drain started siblings so their successes stay reusable.
                for future, item in tuple(pending.items()):
                    if future.cancel():
                        pending.pop(future)
                        collect(future, item)
                while pending and not stopped:
                    collect_finished(
                        in_plan_order(self._finished(completions, pending))
                    )
                    if self._stop_requested():
                        stopped = True

            if stopped:
                # Keep whatever already finished, then abandon the rest.
                collect_finished(
                    in_plan_order(future for future in pending if future.done())
                )
                for future, item in tuple(pending.items()):
                    future.cancel()
                    record_non_reusable(item.work_id, NonReusableReason.CANCELLED)
                pending.clear()
        finally:
            try:
                if executor is not None:
                    if stopped:
                        executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        executor.shutdown(wait=True)
            except Exception as error:  # noqa: BLE001 - retain cleanup failure as cause
                cleanup_error = error

        if stopped and terminal_error is None:
            for item in work[next_index:]:
                record_non_reusable(item.work_id, NonReusableReason.CANCELLED)
            checkpoint_error = self._checkpoint(
                session,
                non_reusable,
                positions,
                terminal_failure=None,
                terminal_work_id=None,
            )
            stop = PipelineStopped("stopped between scheduled items")
            cause = checkpoint_error or cleanup_error
            if cause is not None:
                raise stop from cause
            raise stop

        if terminal_error is None and cleanup_error is not None:
            terminal_error = cleanup_error

        if terminal_error is not None:
            for item in work[next_index:]:
                record_non_reusable(item.work_id, NonReusableReason.UNKNOWN)
            checkpoint_error = self._checkpoint(
                session,
                non_reusable,
                positions,
                terminal_failure=True,
                terminal_work_id=terminal_work_id,
            )
            if checkpoint_error is not None:
                raise terminal_error from checkpoint_error
            if cleanup_error is not None and cleanup_error is not terminal_error:
                raise terminal_error from cleanup_error
            raise terminal_error

        if (
            session.manifest.terminal_failure
            or session.manifest.terminal_failure_work_id is not None
        ):
            checkpoint_error = self._checkpoint(
                session,
                non_reusable,
                positions,
                terminal_failure=False,
                terminal_work_id=None,
            )
            if checkpoint_error is not None:
                raise checkpoint_error
        return tuple(
            results[work_id] for work_id in sorted(results, key=positions.__getitem__)
        )

    def run_levels(
        self,
        levels: Iterable[tuple[ScheduledWork, ...]],
        session: CheckpointSession,
    ) -> tuple[ScheduledResult, ...]:
        results: list[ScheduledResult] = []
        for level in levels:
            results.extend(self.run(level, session))
        positions = {
            work_id: index for index, work_id in enumerate(session.manifest.work_ids)
        }
        return tuple(sorted(results, key=lambda item: positions[item.work_id]))

    def _stop_requested(self) -> bool:
        return self._should_stop is not None and self._should_stop()

    def _finished(
        self,
        completions: SimpleQueue[Future[object]],
        pending: Mapping[Future[object], ScheduledWork],
    ) -> set[Future[object]]:
        """Wait until pending work finishes, or return early for a stop."""
        timeout = None if self._should_stop is None else STOP_POLL_SECONDS
        done: set[Future[object]] = set()
        while not done:
            try:
                future = completions.get(timeout=timeout)
            except Empty:
                if self._stop_requested():
                    break
                continue
            if future in pending:
                done.add(future)
        while True:
            try:
                future = completions.get_nowait()
            except Empty:
                return done
            if future in pending:
                done.add(future)

    @staticmethod
    def _validate_work(
        work: tuple[ScheduledWork, ...], session: CheckpointSession
    ) -> None:
        work_ids = tuple(item.work_id for item in work)
        planned = set(session.manifest.work_ids)
        unavailable = {reference.work_id for reference in session.manifest.completed}
        if (
            len(set(work_ids)) != len(work_ids)
            or not set(work_ids).issubset(planned)
            or set(work_ids) & unavailable
        ):
            raise ValueError("work must be unique planned work without prior state")

    @staticmethod
    def _checkpoint(
        session: CheckpointSession,
        non_reusable: Mapping[str, NonReusableReason],
        positions: Mapping[str, int],
        *,
        terminal_failure: bool | None,
        terminal_work_id: str | None,
    ) -> Exception | None:
        """Record the batch outcome; completed work was recorded as it finished."""
        try:
            session.checkpoint_scheduler_state(
                non_reusable=tuple(
                    NonReusableRef(work_id=work_id, reason=non_reusable[work_id])
                    for work_id in sorted(non_reusable, key=positions.__getitem__)
                ),
                terminal_failure=terminal_failure,
                terminal_failure_work_id=terminal_work_id,
                clear_terminal_failure=(
                    terminal_failure is not None and terminal_work_id is None
                ),
            )
        except Exception as error:  # noqa: BLE001 - caller receives the terminal work error
            return error
        return None
