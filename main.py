import os
import json
import signal
import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List
import datetime
import aiomqtt
from langfuse import get_client
import redis.asyncio as redis
from dotenv import load_dotenv

from agent_graph import HazardMapReduceManager
from memgraph_custom_tools import get_memgraph_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()
langfuse = get_client()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QUEUE_NAME = os.getenv("QUEUE_NAME", "room_pq")

MQTT_BROKER_ADDRESS = os.getenv("MQTT_BROKER_ADDRESS", "localhost")
MQTT_PUBLISH_TOPIC = os.getenv("MQTT_PUBLISH_TOPIC", "emergency/agent/data")


class HazardWorker:
    def __init__(self):
        self.redis: redis.Redis | None = None
        # self.swarm_manager = HazardSwarmManager()
        self.swarm_manager = HazardMapReduceManager()
        self.stop_event = asyncio.Event()

    async def initialize(self) -> None:
        self.redis = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
        )

        memgraph_tools = await get_memgraph_tools()

        await self.swarm_manager.initialize_graph(
            tools=memgraph_tools,
            debug=True,
            print_agent=True,
        )

    async def cleanup(self) -> None:
        if self.redis:
            await self.redis.aclose()

    def stop(self) -> None:
        logger.info("Shutdown signal received. Exiting on next cycle...")
        self.stop_event.set()

    @staticmethod
    def parse_snapshots(
        zrange_data: List[Dict[str, Any]], room: str
    ) -> List[Dict[str, Any]]:
        snapshots = defaultdict(dict)

        for json_str, timestamp in zrange_data:
            timestamp = int(timestamp)
            data = json.loads(json_str)
            data["room"] = room
            data["timestamp"] = datetime.datetime.fromtimestamp(
                timestamp / 1000
            ).strftime("%Y-%m-%d %H:%M:%S")
            snapshots[timestamp] = data

        return [snapshot for _, snapshot in sorted(snapshots.items())]

    async def fetch_and_process_room(self, room: str) -> list[Dict[str, Any]] | None:
        zrange_result = await self.redis.zrange(room + ":ts", 0, -1, withscores=True)

        parsed_data = self.parse_snapshots(zrange_result, room)
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
                    async with aiomqtt.Client(MQTT_BROKER_ADDRESS) as client:
                        for item in processed_data:
                            await client.publish(MQTT_PUBLISH_TOPIC, json.dumps(item))
                            logger.info("[↑] Published: %s", item)
            except Exception as e:
                logger.error("An error occurred: %s", e)
                await asyncio.sleep(2)
            except aiomqtt.MqttError as error:
                print(f"MQTT error: {error}")
            except asyncio.CancelledError:
                print("Async task cancelled")


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
