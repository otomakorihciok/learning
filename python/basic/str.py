import logging
from logging import getLogger, StreamHandler

LOGGER = getLogger(__name__)
LOGGER.setLevel(logging.INFO)

stream_handler = StreamHandler()

# handlerのログレベル設定(ハンドラが出力するエラーメッセージのレベル)
stream_handler.setLevel(logging.INFO)

LOGGER.addHandler(stream_handler)


class E:

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __str__(self):
    return 'This is sum.'


LOGGER.info(f'content is {Exception()}')

print('Message is', E())
