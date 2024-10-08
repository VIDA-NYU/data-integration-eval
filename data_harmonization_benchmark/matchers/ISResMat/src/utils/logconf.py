import datetime
import logging

import pytz

basic_format = '%(asctime)s %(levelname)s %(message)s'

class Formatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp, pytz.UTC)
        return dt.astimezone(pytz.timezone('Asia/Shanghai'))

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.strftime('%Y-%m-%d %H:%M:%S')
            # s = dt.isoformat(sep=' ', timespec='milliseconds')
        return s


def get_simple_print_logger(name, level=logging.INFO, propagate=False):
    formatter = Formatter(basic_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)

    logger.propagate = propagate

    return logger