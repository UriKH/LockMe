"""
The application main function in which the user chooses to use the command line interface or the Tkinter UI
"""

import terminal_ui.main as terminal_main
import tk_ui.main as tk_main
from utils.logger import Logger
from utils.messages import Messages as msg


def main():
    Logger(msg.Info.hello_world, Logger.message).log()

    while True:
        Logger(msg.Info.choose_ui + ' ', Logger.message, new_line=False).log()
        stay_in_terminal = input()
        if not stay_in_terminal.isalpha():
            continue

        stay_in_terminal = stay_in_terminal.lower()
        if stay_in_terminal == 'y':
            terminal_main.main()
            break
        elif stay_in_terminal == 'n':
            tk_main.main()
            break


if __name__ == '__main__':
    main()
