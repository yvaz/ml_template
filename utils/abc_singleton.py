from abc import ABCMeta

class ABCSingleton(ABCMeta):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(ABCSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]