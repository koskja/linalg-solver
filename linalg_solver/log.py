from .fmt import pcformat


class Logger:
    accum: list[str]
    level_limit: int = 0
    _auto_print: bool = False

    def __init__(self, accum: list[str] = None, level_limit: int = 0):
        self.accum = accum if accum is not None else []
        self.level_limit = level_limit

    def log(self, message: str, level=0):
        if level > self.level_limit:
            return
        self.accum.append(message)
        if self._auto_print:
            print(message)

    def __str__(self):
        return "\n".join(self.accum)


def push_logger(logger=None):
    global current_logger, logger_stack
    if logger is None:
        logger = Logger()
    logger_stack.append(logger)
    current_logger = logger_stack[-1]


def pop_logger() -> Logger:
    global current_logger, logger_stack
    if len(logger_stack) == 0:
        raise ValueError("No logger to pop")
    ret = logger_stack.pop()
    current_logger = logger_stack[-1]
    return ret


def log(message: str, *args):
    raw_log(pcformat(message, *args))


def ignore_log(f):
    with nest_logger():
        return f()


def raw_log(message: str):
    global current_logger
    current_logger.log(message)


class LoggerGuard:
    def __init__(self, logger=None, append_logs: list[str] = None):
        self.logger = logger
        self.append_logs = append_logs

    def __enter__(self):
        global current_logger
        push_logger(self.logger)
        return current_logger

    def get_logger(self):
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        lg = pop_logger()
        if self.append_logs is not None:
            if len(lg.accum) > 0:
                self.append_logs.append(str(lg))
        return False


def nest_logger():
    return LoggerGuard()


def nest_appending_logger(logs_list: list[str]):
    return LoggerGuard(append_logs=logs_list)


def capture_logs(f) -> str:
    with nest_logger() as lg:
        f()
    return str(lg)


current_logger = None
logger_stack = []
global_logger = Logger()
global_logger._auto_print = True
push_logger(global_logger)


def poorly_formatted(a, b):
    return a + b
