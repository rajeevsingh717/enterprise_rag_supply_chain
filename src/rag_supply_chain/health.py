"""Health check for the infrastructure stack.

Probes each service (Redpanda, Qdrant, Postgres, Redis) independently and prints
a per-service verdict. Exits non-zero if any probe fails, so `make health` is a
usable gate. Each probe is isolated: a missing client library or a down service
is reported for that row only, never crashes the whole run.

    python -m rag_supply_chain.health      # or: rag-health
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from rag_supply_chain.config import settings

console = Console()


def check_redpanda() -> tuple[bool, str]:
    try:
        from confluent_kafka.admin import AdminClient

        admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
        md = admin.list_topics(timeout=5)
        return True, f"{len(md.brokers)} broker(s), {len(md.topics)} topic(s)"
    except Exception as e:  # noqa: BLE001 - report any failure per-service
        return False, str(e).splitlines()[0]


def check_qdrant() -> tuple[bool, str]:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, timeout=5)
        collections = client.get_collections().collections
        return True, f"reachable, {len(collections)} collection(s)"
    except Exception as e:  # noqa: BLE001
        return False, str(e).splitlines()[0]


def check_postgres() -> tuple[bool, str]:
    try:
        import psycopg

        with psycopg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            dbname=settings.postgres_db,
            connect_timeout=5,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
        return True, version.split(",")[0]
    except Exception as e:  # noqa: BLE001
        return False, str(e).splitlines()[0]


def check_redis() -> tuple[bool, str]:
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=5)
        client.ping()
        return True, "PONG"
    except Exception as e:  # noqa: BLE001
        return False, str(e).splitlines()[0]


CHECKS = {
    "Redpanda (Kafka)": check_redpanda,
    "Qdrant": check_qdrant,
    "Postgres": check_postgres,
    "Redis": check_redis,
}


def main() -> int:
    table = Table(title="RAG Supply Chain — Infra Health")
    table.add_column("Service", style="bold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    all_ok = True
    for name, check in CHECKS.items():
        ok, detail = check()
        all_ok = all_ok and ok
        status = "[green]✓ up[/green]" if ok else "[red]✗ down[/red]"
        table.add_row(name, status, detail)

    console.print(table)
    if all_ok:
        console.print("[bold green]All services healthy.[/bold green]")
        return 0
    console.print("[bold red]One or more services are unreachable.[/bold red]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
