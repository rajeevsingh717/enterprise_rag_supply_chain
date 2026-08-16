"""Celery application: Redis-backed distributed task queue (§3.B).

Start a worker with:

    celery -A rag_supply_chain.workers.celery_app worker --loglevel=info
"""

from __future__ import annotations

from celery import Celery

from rag_supply_chain.config import settings

celery_app = Celery(
    "rag_supply_chain",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["rag_supply_chain.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
)
