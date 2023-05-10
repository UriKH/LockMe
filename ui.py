# all other classes like camera_runner should inherit from this class.
# this class will define the outputs which are not from the terminal
import cv2 as cv
import os
import pandas as pd

from logger import Logger
from messages import Messages as msg
from keys import KeyMap
from initialize import Init
from database import Database

commands = {KeyMap.exit_cmd: KeyMap.exit, KeyMap.add_cmd: KeyMap.add, KeyMap.remove_cmd: KeyMap.remove,
            KeyMap.delete_cmd: KeyMap.delete, KeyMap.trash_cmd: KeyMap.trash,
            KeyMap.log_off_cmd: KeyMap.log_off, KeyMap.show_cmd: KeyMap.show, KeyMap.lock_cmd: KeyMap.lock,
            KeyMap.unlock_cmd: KeyMap.unlock}

cmd_param = {KeyMap.add_cmd: 1, KeyMap.remove_cmd: 1, KeyMap.delete_cmd: 0, KeyMap.trash_cmd: 1,
             KeyMap.log_off_cmd: 0, KeyMap.show_cmd: 0, KeyMap.exit_cmd: 0, KeyMap.lock_cmd: 1, KeyMap.unlock_cmd: 1}


def parse_n_call(cmd, line, user):
    """
    Parse and execute a command
    :param cmd: the command to be executed
    :param line: the whole line of input from user
    :param user: Object of User class
    :return: None if execution for the current user should continue, else, returns True if the program should exit or
        False if the user logged-off
    """
    line = line.rstrip()
    line = line.lstrip()
    pieces = [char for char in line.split(' ') if char != '']
    if len(pieces) > 1:
        pieces = pieces[1:]
    else:
        pieces = []

    if len(pieces) < cmd_param[cmd]:
        Logger(msg.Errors.missing_param, Logger.warning).log()
        return None
    elif len(pieces) > cmd_param[cmd]:
        Logger(msg.Errors.to_many_params, Logger.warning).log()
        return None

    try:
        if cmd == KeyMap.add_cmd:
            Init.database.add_file(*pieces, user)
        elif cmd == KeyMap.remove_cmd:
            Init.database.remove_file(*pieces, user)
        elif cmd == KeyMap.lock_cmd:
            Init.database.lock_file(*pieces, user)
        elif cmd == KeyMap.unlock_cmd:
            Init.database.unlock_file(*pieces, user)
        elif cmd == KeyMap.exit_cmd:
            Logger(msg.Info.exiting, Logger.message).log()
            return True
        elif cmd == KeyMap.log_off_cmd:
            Logger(msg.Info.logging_off, Logger.message).log()
            Init.database.lock_all_files(user.uid)
            return False
        elif cmd == KeyMap.show_cmd:
            data = Init.database.fetch_user_data(user.uid)
            present_data(data)
        elif cmd == KeyMap.trash_cmd:
            delete_file(*pieces)
        elif cmd == KeyMap.delete_cmd:
            Init.database.delete_user(user.uid)
            Logger(msg.Info.logging_off, Logger.message).log()
            return False
        return None
    except Exception as e:
        Logger(e, Logger.inform).log()
        Logger(msg.Info.back_to_routine, Logger.info).log()


def present_data(data):
    """
    Present the data of the user
    :param data: a dictionary representing the user's data
    """
    data = data.copy()
    del data['file']
    del data['user_id']
    data['file_state'] = ['locked' if state == Database.file_state_locked else 'open' for state in data['file_state']]

    df = pd.DataFrame(data)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    Logger(df, Logger.message).log()


def delete_file(path):
    """
    Delete a file from disk
    :param path: the path to the file
    """
    try:
        os.remove(path)
        Logger(msg.Info.file_deleted + f' {path}', Logger.warning).log()
    except Exception as e:
        Logger(e, Logger.inform).log()
        Logger(msg.Info.back_to_routine, Logger.info).log()


def add_file(path, user):
    """
    Add a file to the user
    :param path: path to the file to remove
    :param user: a User object
    """
    try:
        Init.database.add_file(path, user)
    except Exception as e:
        Logger(e, Logger.inform).log()
        Logger(msg.Info.back_to_routine, Logger.info).log()


def remove_file(path, user):
    """
    Remove a file from the user's account
    :param path: path to the file to remove
    :param user: a User object
    """
    try:
        Init.database.remove_file(path, user)
    except Exception as e:
        Logger(e, Logger.inform).log()
        Logger(msg.Info.back_to_routine, Logger.info).log()


def present_menu(user) -> bool:
    """
    Present a menu of options to the user and execute the user's requests
    :param user: An object of User class
    :return: True if program should exit, False if log-off
    """
    Logger(msg.Info.menu, Logger.message).log()

    while True:
        line = input('>>> ')
        line = line.rstrip()
        line = line.lstrip()
        command = line.split(' ')[0]
        if command.lower() not in commands.keys() and command.lower() not in commands.values():
            Logger(msg.Errors.invalid_cmd, Logger.warning).log()
            continue

        if len(command) == 1:
            command = list(commands.keys())[list(commands.values()).index(command)]

        fully_exit = parse_n_call(command, line, user)
        if fully_exit is None:
            continue
        if fully_exit:
            return True
        else:
            return False


def continue_to_system() -> bool:
    """
    Ask the user to join the system
    :return: True if the user wants to continue to the system
    """
    Logger(msg.Requests.want_to_join, level=Logger.message).log()
    while True:
        key = cv.waitKey()
        if key == ord(KeyMap.yes):
            return True
        elif key == ord(KeyMap.no):
            return False
