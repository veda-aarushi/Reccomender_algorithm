from celery_app import celery
from content_engine import ContentEngine

@celery.task(name="tasks.train_task")
def train_task(csv_path: str):
    engine = ContentEngine()
    return engine.train(csv_path)
