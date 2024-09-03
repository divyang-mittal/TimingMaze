import logging
import unicodedata
import re


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '-', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


class MainLoggingFilter(logging.Filter):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def filter(self, record):
        if record.name == self.name:
            return True
        else:
            return False


class PlayerLoggingFilter(logging.Filter):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def filter(self, record):
        if self.name in record.name or record.name == __name__:
            return True
        else:
            return False


def isiterable(obj):
    try:
        iterator = iter(obj)
    except TypeError as te:
        return False
    return True


def count_iterable(i):
    return sum(1 for e in i)
