import constants as C


def get_device():
    return C.DEVICE


def to_device(*args):
    if len(args) > 1:
        return [a.to(get_device()) for a in args]
    else:
        return args[0].to(get_device())
