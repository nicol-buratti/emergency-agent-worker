import os
import json
import signal
import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List

import redis.asyncio as redis
from dotenv import load_dotenv
from agent_swarm_prebuilts import HazardSwarmManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QUEUE_NAME = os.getenv("QUEUE_NAME", "room_pq")
PUBLISH_CHANNEL = os.getenv("PUBLISH_CHANNEL", "sensors_processed")


class HazardWorker:
    def __init__(self):
        self.redis: redis.Redis | None = None
        self.swarm_manager = HazardSwarmManager()
        self.stop_event = asyncio.Event()

    async def initialize(self) -> None:
        self.redis = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
        )
        # mcp_client_thingsboard = MultiServerMCPClient(
        #     {
        #         "thingsboard": {"url": "http://localhost:8000/sse", "transport": "sse"},
        #         # "ddgs": {"command": "ddgs", "args": ["mcp"], "transport": "stdio"},
        #     }
        # )

        # thingsboard_tools = await mcp_client_thingsboard.get_tools()
        await self.swarm_manager.initialize_graph(debug=True, print_agent=True)

    async def cleanup(self) -> None:
        if self.redis:
            await self.redis.aclose()

    def stop(self) -> None:
        logger.info("Shutdown signal received. Exiting on next cycle...")
        self.stop_event.set()

    @staticmethod
    def parse_snapshots(mrange_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        snapshots = defaultdict(dict)

        for series in mrange_data:
            for _, (labels, datapoints) in series.items():
                sensor_type = labels.get("type")
                room_name = labels.get("room")

                for timestamp, value in datapoints:
                    snapshots[timestamp]["timestamp"] = timestamp
                    snapshots[timestamp]["room"] = room_name
                    snapshots[timestamp][sensor_type] = value

        return [snapshot for _, snapshot in sorted(snapshots.items())]

    async def fetch_and_process_room(self, room: str) -> Dict[str, Any] | None:
        mrange_result = await self.redis.ts().mrange(
            from_time="-",
            to_time="+",
            with_labels=True,
            filters=[f"room={room}"],
            aggregation_type="avg",
            bucket_size_msec=60000,
        )

        parsed_data = self.parse_snapshots(mrange_result)
        if not parsed_data:
            logger.warning("No data associated with room %s.", room)
            return None

        payload = {"room": room, "sensor_data": parsed_data}
        return await self.swarm_manager.process_data(payload)

    async def run(self) -> None:
        logger.info("Worker listening on priority queue '%s'...", QUEUE_NAME)

        while not self.stop_event.is_set():
            try:
                queue_result = await self.redis.bzpopmin(QUEUE_NAME, timeout=5)
                if not queue_result:
                    continue

                _, room, score = queue_result
                logger.info("[↓] Extracted (Priority: %s): %s", score, room)

                start_time = time.perf_counter()
                processed_data = await self.fetch_and_process_room(room)
                elapsed_time = time.perf_counter() - start_time

                logger.info(
                    "[⏱️] Processing took %.4fs for room: %s", elapsed_time, room
                )

                if processed_data:
                    await self.redis.publish(
                        PUBLISH_CHANNEL, json.dumps(processed_data)
                    )
                    logger.info("[↑] Published: %s", processed_data)

            except Exception as e:
                logger.error("An error occurred: %s", e)
                await asyncio.sleep(2)


async def main():
    worker = HazardWorker()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker.stop)

    try:
        await worker.initialize()
        await worker.run()
    finally:
        await worker.cleanup()
        logger.info("Worker gracefully stopped.")


if __name__ == "__main__":
    asyncio.run(main())
