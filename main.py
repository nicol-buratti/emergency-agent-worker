import signal
import sys
import asyncio
import redis.asyncio as redis  # <-- Import fondamentale per l'uso asincrono

from agent import build_graph, call_agent

# Connessione a Redis ASINCRONA
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

QUEUE_NAME = "room_pq"
PUBLISH_CHANNEL = "sensors_processed"

# Variabile globale
in_esecuzione = True


def spegnimento_sicuro(signum, frame):
    """Cattura il segnale SIGTERM (Docker) o SIGINT (Ctrl+C)."""
    global in_esecuzione
    print(
        f"\n[*] Ricevuto segnale di spegnimento ({signum}). Uscita al prossimo ciclo..."
    )
    in_esecuzione = False


# Registra i gestori di segnale
signal.signal(signal.SIGTERM, spegnimento_sicuro)
signal.signal(signal.SIGINT, spegnimento_sicuro)


async def processa_sensore(room):
    """Logica di elaborazione del task."""
    # Essendo r diventato asincrono, dobbiamo usare await
    result = await r.get(room)

    if result is None:
        print(f"[!] Nessun dato associato alla stanza {room}.")
        return None

    # Questa chiamata ora bloccherà l'esecuzione fino alla risposta dell'LLM
    result = await call_agent(result)

    dato_elaborato = f"PROCESSED_{room}: {result}"
    return dato_elaborato


async def main():
    await build_graph()
    print(f"[*] Worker in ascolto sulla priority queue (ZSET) '{QUEUE_NAME}'...")
    print(f"[*] I risultati verranno pubblicati sul canale '{PUBLISH_CHANNEL}'.")
    print("[*] Premi Ctrl+C per uscire.\n")

    while in_esecuzione:
        try:
            # await su bzpopmin garantisce che il loop non impazzisca
            risultato = await r.bzpopmin(QUEUE_NAME, timeout=5)

            if risultato:
                chiave_coda, room, score = risultato
                print(f"[↓] Estratto (Priorità: {score}): {room}")

                # 1. Elabora il dato (il loop si ferma qui finché call_agent non finisce)
                dato_elaborato = await processa_sensore(room)

                # 2. Fai il publish sul canale Pub/Sub
                if dato_elaborato:
                    await r.publish(PUBLISH_CHANNEL, dato_elaborato)
                    print(f"[↑] Pubblicato: {dato_elaborato}\n")

        except Exception as e:
            print(f"[!] Si è verificato un errore: {e}")
            # time.sleep(2) bloccava tutto il thread. Usa asyncio.sleep
            await asyncio.sleep(2)

    print("[*] Worker arrestato in modo sicuro.")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
