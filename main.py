import terminal_ui.main as terminal_main
import tk_ui.main as tk_main
from logger import Logger
from messages import Messages as msg


def main():
    Logger(msg.Info.hello_world, Logger.message).log()

    while True:
        Logger(msg.Info.choose_ui + ' ', Logger.message, new_line=False).log()
        stay_terminal = input()
        if not stay_terminal.isalpha():
            continue

        stay_terminal = stay_terminal.lower()
        if stay_terminal == 'y':
            terminal_main.main()
            break
        elif stay_terminal == 'n':
            tk_main.main()
            break


if __name__ == '__main__':
    main()
