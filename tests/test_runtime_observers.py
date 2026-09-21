from summarizer.runtime.observers import RuntimeObserver, StageEvent, StageName


def test_observer_emits_and_checks_cancel():
    events: list[StageEvent] = []
    cancelled = {"value": False}
    observer = RuntimeObserver(
        on_stage=events.append,
        should_cancel=lambda: cancelled["value"],
    )
    observer.emit(StageEvent(StageName.PREPARING, "active"))
    assert events[0].stage == StageName.PREPARING
    assert not observer.cancelled()
    cancelled["value"] = True
    assert observer.cancelled()
