import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler
import os


def setup(log_directory: str):
    format = '[%(asctime)s] %(levelname)s \t [%(filename)s %(lineno)d] %(message)s'

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(format))

    # server
    server_logger = logging.getLogger("server")
    server_logger.addHandler(stream_handler)
    server_info_file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_directory, 'server.log'),
        encoding='utf-8',
        when='D',
        interval=1,
        backupCount=7
    )
    server_info_file_handler.setLevel(logging.INFO)
    server_info_file_handler.setFormatter(logging.Formatter(format))
    server_logger.addHandler(server_info_file_handler)
    # server_debug_file_handler = TimedRotatingFileHandler(
    #     filename=os.path.join(log_directory, 'server_debug.log'),
    #     encoding='utf-8',
    #     when='D',
    #     interval=1,
    #     backupCount=7
    # )
    # server_debug_file_handler.setLevel(logging.DEBUG)
    # server_debug_file_handler.setFormatter(logging.Formatter(format))
    # server_logger.addHandler(server_debug_file_handler)

    # apscheduler
    apscheduler_logger = logging.getLogger("apscheduler")
    apscheduler_logger.addHandler(stream_handler)
    apscheduler_file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_directory, 'apscheduler.log'),
        encoding='utf-8',
        when='D',
        interval=1,
        backupCount=7
    )
    apscheduler_file_handler.setLevel(logging.INFO)
    apscheduler_file_handler.setFormatter(logging.Formatter(format))
    apscheduler_logger.addHandler(apscheduler_file_handler)

    # elasticsearch
    es_logger = logging.getLogger("elasticsearch")
    es_file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_directory, 'elasticsearch.log'),
        encoding='utf-8',
        when='D',
        interval=1,
        backupCount=7
    )
    es_file_handler.setLevel(logging.DEBUG)
    es_file_handler.setFormatter(logging.Formatter(format))
    es_logger.addHandler(es_file_handler)

    debug_file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_directory, 'debug.log'),
        encoding='utf-8',
        when='D',
        interval=1,
        backupCount=7
    )
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(logging.Formatter(format))

    info_file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_directory, 'info.log'),
        encoding='utf-8',
        when='D',
        interval=1,
        backupCount=7
    )
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(logging.Formatter(format))

    error_file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_directory, 'error.log'),
        encoding='utf-8',
        when='D',
        interval=1,
        backupCount=7
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(logging.Formatter(format))

    logging.basicConfig(
        level=logging.DEBUG,
        datefmt='%a, %d %b %Y %H:%M:%S',
        handlers=[
            debug_file_handler,
            info_file_handler,
            error_file_handler,
        ]
    )
    return
