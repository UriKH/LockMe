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
            print(f'>  {self.msg}')
        if self.level == Logger.info:           # info - green
            print(f'\033[92m[INFO] {self.msg}\033[92m')
        elif self.level == Logger.warning:      # warning - yellow
            print(f'\033[93m[WARNING] {self.msg}\033[93m')
        elif self.level == Logger.exception:    # exception - red
            raise Exception(f'\033[91m[EXCEPT] {self.msg}\033[91m')
        elif self.level == Logger.error:        # fatal error - red
            print('\033[91m[ERROR] fatal error - exiting\033[91m')
            exit(1)