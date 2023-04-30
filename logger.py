"""
Logging for terminal UI and debugging
"""


class Logger:
    message = 0
    info = 1
    warning = 2
    exception = 3
    error = 4

    def __init__(self, msg, level=message):
        self.level = level if level in range(Logger.error + 1) else Logger.info
        self.msg = msg

    def log(self):
        if self.level == Logger.message:        # message
            print(f'> {self.msg}')
        if self.level == Logger.info:           # info
            print(f'[INFO] {self.msg}')
        elif self.level == Logger.warning:      # warning
            print(f'[WARNING] {self.msg}')
        elif self.level == Logger.exception:    # exception
            raise Exception(f'[EXCEPT] {self.msg}')
        elif self.level == Logger.error:        # fatal error
            print('[ERROR] fatal error - exiting')
            exit(1)
