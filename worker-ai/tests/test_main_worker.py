import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.main as main
from src.main import HazardWorker


@pytest.fixture(autouse=True)
def _stub_swarm_manager_class(monkeypatch):
    """HazardWorker() builds a real HazardMapReduceManager in __init__,
    which in turn requires LLM_API_KEY. These tests care about the worker
    loop, Redis interaction, and snapshot parsing, not the real agent
    swarm, so the class is stubbed out here at construction time.
    Individual tests still replace `worker.swarm_manager` with their own
    mock when they need to assert on it.
    """
    monkeypatch.setattr(main, "HazardMapReduceManager", MagicMock())


class TestParseSnapshots:
    """HazardWorker.parse_snapshots is pure logic (no I/O) and is the piece
    most likely to have subtle ordering/dedup bugs, so it's tested in
    isolation from Redis and the agent swarm."""

    def test_returns_empty_list_for_no_data(self):
        assert HazardWorker.parse_snapshots([], "room-101") == []

    def test_parses_single_snapshot(self):
        zrange_data = [(json.dumps({"temp": 22.5}), 1_700_000_000_000)]

        result = HazardWorker.parse_snapshots(zrange_data, "room-101")

        assert len(result) == 1
        assert result[0]["temp"] == 22.5
        assert result[0]["room"] == "room-101"
        assert "timestamp" in result[0]

    def test_orders_snapshots_chronologically_regardless_of_input_order(self):
        """Redis zrange is expected to already be ordered, but the code
        re-sorts by timestamp, so out-of-order input must still come back
        sorted."""
        newer = (json.dumps({"temp": 30}), 1_700_000_020_000)
        older = (json.dumps({"temp": 20}), 1_700_000_000_000)
        middle = (json.dumps({"temp": 25}), 1_700_000_010_000)

        result = HazardWorker.parse_snapshots([newer, older, middle], "room-101")

        assert [s["temp"] for s in result] == [20, 25, 30]

    def test_deduplicates_snapshots_with_identical_timestamp(self):
        """Snapshots are keyed by timestamp in a dict, so two entries with
        the exact same timestamp collapse into one (last write wins)."""
        first = (json.dumps({"temp": 20}), 1_700_000_000_000)
        second = (json.dumps({"temp": 99}), 1_700_000_000_000)

        result = HazardWorker.parse_snapshots([first, second], "room-101")

        assert len(result) == 1
        assert result[0]["temp"] == 99

    def test_injects_room_and_readable_timestamp_into_every_snapshot(self):
        zrange_data = [
            (json.dumps({"temp": 22}), 1_700_000_000_000),
            (json.dumps({"temp": 23}), 1_700_000_060_000),
        ]

        result = HazardWorker.parse_snapshots(zrange_data, "room-202")

        assert all(s["room"] == "room-202" for s in result)
        assert all(isinstance(s["timestamp"], str) for s in result)

    def test_raises_on_malformed_json_payload(self):
        zrange_data = [("{not valid json", 1_700_000_000_000)]

        with pytest.raises(json.JSONDecodeError):
            HazardWorker.parse_snapshots(zrange_data, "room-101")


class TestFetchAndProcessRoom:
    @pytest.fixture
    def worker(self):
        worker = HazardWorker()
        worker.redis = AsyncMock()
        worker.swarm_manager = AsyncMock()
        return worker

    @pytest.mark.asyncio
    async def test_returns_none_when_room_has_no_snapshots(self, worker):
        worker.redis.zrange.return_value = []

        result = await worker.fetch_and_process_room("room-101")

        assert result is None
        worker.swarm_manager.process_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_queries_the_correct_redis_key(self, worker):
        worker.redis.zrange.return_value = []

        await worker.fetch_and_process_room("room-101")

        worker.redis.zrange.assert_awaited_once_with(
            "room-101:ts", 0, -1, withscores=True
        )

    @pytest.mark.asyncio
    async def test_passes_parsed_snapshots_to_swarm_manager(self, worker):
        worker.redis.zrange.return_value = [
            (json.dumps({"temp": 22.5}), 1_700_000_000_000)
        ]
        worker.swarm_manager.process_data.return_value = [
            {"room": "room-101", "danger": "none"}
        ]

        result = await worker.fetch_and_process_room("room-101")

        worker.swarm_manager.process_data.assert_awaited_once()
        call_payload = worker.swarm_manager.process_data.await_args.args[0]
        assert call_payload["room"] == "room-101"
        assert len(call_payload["sensor_data"]) == 1
        assert result == [{"room": "room-101", "danger": "none"}]


class TestStop:
    def test_stop_sets_the_stop_event(self):
        worker = HazardWorker()

        assert not worker.stop_event.is_set()
        worker.stop()

        assert worker.stop_event.is_set()


class TestRunLoop:
    """Exercises a couple of iterations of the main run() loop against a
    faked Redis client, without needing a real broker or LLM."""

    @pytest.fixture
    def worker(self):
        worker = HazardWorker()
        worker.redis = AsyncMock()
        worker.swarm_manager = AsyncMock()
        return worker

    @pytest.mark.asyncio
    async def test_stops_when_stop_event_is_set_before_first_iteration(self, worker):
        worker.stop_event.set()

        await worker.run()

        worker.redis.bzpopmin.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_queue_result_does_not_call_process_room(self, worker):
        """bzpopmin returning None/empty (timeout) should just loop again,
        not attempt to process a room."""
        call_count = 0

        async def fake_bzpopmin(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                worker.stop_event.set()
            return None

        worker.redis.bzpopmin.side_effect = fake_bzpopmin

        await worker.run()

        assert call_count == 2
        worker.swarm_manager.process_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_exception_during_processing_does_not_crash_the_loop(
        self, worker, monkeypatch
    ):
        """The run loop must survive a single bad iteration (e.g. a Memgraph
        or LLM hiccup) and keep listening on the queue rather than dying."""
        call_count = 0

        async def fake_bzpopmin(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (b"room_pq", "room-101", 1.0)
            worker.stop_event.set()
            return None

        worker.redis.bzpopmin.side_effect = fake_bzpopmin
        worker.redis.zrange.side_effect = RuntimeError("Redis exploded")

        async def no_sleep(_seconds):
            return None

        monkeypatch.setattr(asyncio, "sleep", no_sleep)

        # Should not raise.
        await worker.run()

        assert call_count == 2
