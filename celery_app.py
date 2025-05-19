from celery import Celery
import config

celery = Celery(
    "tasks",
    broker=config.REDIS_URL,
    backend=config.REDIS_URL,
)
celery.conf.task_routes = {"tasks.train_task": {"queue": "train"}}
