"""
Logging for terminal UI and debugging
"""
import time


class Logger:
    message = 0
    info = 1
    warning = 2
    inform = 3
    exception = 4
    error = 5

    class Colors:
        white = '\033[0m'
        red = '\033[91m'
        green = '\033[92m'
        yellow = '\033[93m'

    def __init__(self, msg, level=message, new_line=True):
        self.level = level if level in range(Logger.error + 1) else Logger.info
        self.msg = msg
        if new_line:
            self.end = f'{Logger.Colors.white}\n'
        else:
            self.end = f'{Logger.Colors.white} '

    def time_it(self, func):
        """
        Decorator to log the processing time of a function
        :param func: the function to time
        :return: the wrapped function
        """
        def wrapper(*args, **kwarg):
            start = time.time()
            result = func(*args, **kwarg)
            end = time.time()
            Logger(self.msg + f' - in {(end - start):.3f} seconds', self.level).log()
            return result
        return wrapper

    def log(self, func=None, msg_prefix='> '):
        """
        Log a message with it's logging level to the standard output
        :param func: function the logger is called from
        :param msg_prefix: the message level prefix for printing
        """
        if self.level == Logger.message:        # message
            print(f'{Logger.Colors.white}{msg_prefix}{self.msg}', end=self.end)
            return
        elif func is None:
            if self.level == Logger.info:           # info - green
                print(f'{Logger.Colors.green}[INFO] {self.msg}', end=self.end)
            elif self.level == Logger.warning:      # warning - yellow
                print(f'{Logger.Colors.yellow}[WARNING] {self.msg}', end=self.end)
            elif self.level == Logger.inform:       # does not raise exception - yellow
                print(f'{Logger.Colors.red}[EXCEPT/ERROR] {self.msg}', end=self.end)
            elif self.level == Logger.exception:    # exception - red
                raise Exception(f'{Logger.Colors.red}[EXCEPT] {self.msg}{self.end}')
            elif self.level == Logger.error:        # fatal error - red
                print(f'{Logger.Colors.red}[ERROR] {self.msg}\n\terror - exiting', end=self.end)
                exit(1)
            return
        if self.level == Logger.info:           # info - green
            print(f'{Logger.Colors.green}[INFO] {self.msg} in {func.__name__}', end=self.end)
        elif self.level == Logger.warning:      # warning - yellow
            print(f'{Logger.Colors.yellow}[WARNING] {self.msg} in {func.__name__}', end=self.end)
        elif self.level == Logger.inform:       # does not raise exception - yellow
            print(f'{Logger.Colors.red}[EXCEPT/ERROR] {self.msg} in {func.__name__}', end=self.end)
        elif self.level == Logger.exception:    # exception - red
            raise Exception(f'{Logger.Colors.red}[EXCEPT] {self.msg} in {func.__name__}{self.end}')
        elif self.level == Logger.error:        # fatal error - red
            print(f'{Logger.Colors.red}[ERROR] {self.msg} in {func.__name__}\n\terror - exiting', end=self.end)
            exit(1)
