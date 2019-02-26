"""Celery."""

from celery import Celery

app = Celery('tasks', broker='pyamqp://user1@localhost//')


@app.task
def add(x, y):
  return x + y
