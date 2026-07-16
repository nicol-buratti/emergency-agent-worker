import json
import signal
import sys
import asyncio
import logging
import redis.asyncio as redis  # <-- Essential import for asynchronous use
from dotenv import load_dotenv
from agent import build_graph, call_agent

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

# ASYNCHRONOUS Redis Connection
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

QUEUE_NAME = "room_pq"
PUBLISH_CHANNEL = "sensors_processed"

# Global variable
is_running = True


def graceful_shutdown(signum, frame):
    """Catches the SIGTERM (Docker) or SIGINT (Ctrl+C) signal."""
    global is_running
    logger.info("Received shutdown signal (%s). Exiting on the next cycle...", signum)
    is_running = False


# Register signal handlers
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)


async def process_sensor(room):
    """Task processing logic."""
    result = await r.hgetall(room)
    # result = json.loads(result) if result else None
    result["room"] = room

    if result is None:
        logger.warning("No data associated with room %s.", room)
        return None

    # This call will now block execution until the LLM responds
    result = await call_agent(result)
    return result


async def main():
    logger.info("Building graph...")
    await build_graph()
    logger.info("Worker listening on priority queue (ZSET) '%s'...", QUEUE_NAME)
    logger.info("Results will be published on channel '%s'.", PUBLISH_CHANNEL)
    logger.info("Press Ctrl+C to exit.")

    while is_running:
        try:
            # await on bzpopmin ensures the loop doesn't block synchronously
            queue_result = await r.bzpopmin(QUEUE_NAME, timeout=5)

            if queue_result:
                _, room, score = queue_result
                logger.info("[↓] Extracted (Priority: %s): %s", score, room)

                # 1. Process the data (the loop pauses here until call_agent finishes)
                processed_data = await process_sensor(room)

                # 2. Publish to the Pub/Sub channel
                if processed_data:
                    await r.publish(PUBLISH_CHANNEL, json.dumps(processed_data))
                    logger.info("[↑] Published: %s", processed_data)

        except Exception as e:
            logger.error("An error occurred: %s", e)
            # time.sleep(2) used to block the whole thread. Use asyncio.sleep
            await asyncio.sleep(2)

    logger.info("Worker gracefully stopped.")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
