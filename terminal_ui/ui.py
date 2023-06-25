import os
import pandas as pd

from utils.logger import Logger
from utils.messages import Messages as msg
from utils.initialize import Init
from database import Database
from user_login import User

from . keys import KeyMap


def parse_n_call(cmd: str, line: str, user: User):
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

    if len(pieces) < KeyMap.cmd_param[cmd]:
        Logger(msg.Errors.missing_param, Logger.warning).log()
        return None
    elif len(pieces) > KeyMap.cmd_param[cmd]:
        Logger(msg.Errors.to_many_params, Logger.warning).log()
        return None

    try:
        match cmd:
            case KeyMap.add_cmd:
                Init.database.add_file(*pieces, user)
            case KeyMap.remove_cmd:
                Init.database.remove_file(*pieces, user)
            case KeyMap.lock_cmd:
                Init.database.lock_file(*pieces, user)
            case KeyMap.unlock_cmd:
                Init.database.unlock_file(*pieces, user)
            case KeyMap.exit_cmd:
                Logger(msg.Info.exiting, Logger.message).log()
                return True
            case KeyMap.log_off_cmd:
                Logger(msg.Info.logging_off, Logger.message).log()
                Init.database.lock_all_files(user.uid)
                return False
            case KeyMap.show_cmd:
                data = Init.database.fetch_user_data(user.uid)
                present_data(data)
            case KeyMap.trash_cmd:
                delete_file(*pieces, user)
            case KeyMap.delete_cmd:
                Init.database.delete_user(user.uid)
                Logger(msg.Info.logging_off, Logger.message).log()
                return False
            case KeyMap.recover_cmd:
                Init.database.recover_file(*pieces, user)
            case KeyMap.unlock_all_cmd:
                Init.database.unlock_all_files(user.uid)
            case KeyMap.lock_all_cmd:
                Init.database.lock_all_files(user.uid)
            case _:
                return None
    except Exception as e:
        Logger(e, Logger.inform).log()
        Logger(msg.Info.back_to_routine, Logger.info).log()


def present_data(data: dict):
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

    Logger(df, Logger.message).log(msg_prefix=' ')


def delete_file(path: str, user: User):
    """
    Delete a file from disk
    :param path: the path to the file
    :param user: a User object
    """
    try:
        remove_file(path, user)
        os.remove(path)
        Logger(msg.Info.file_deleted + f' {path}', Logger.warning).log()
    except Exception as e:
        Logger(e, Logger.inform).log()
        Logger(msg.Info.back_to_routine, Logger.info).log()


def add_file(path: str, user: User):
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


def remove_file(path: str, user: User):
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


def present_menu(user: User):
    """
    Present a menu of options to the user and execute the user's requests
    :param user: An object of User class
    :return: True if program should exit, False if log-off
    """
    Logger('\n' + msg.Info.barrier).log(msg_prefix=' ')
    Logger(msg.Info.menu, Logger.message).log(msg_prefix=' ')
    Logger(msg.Info.barrier + '\n').log(msg_prefix=' ')

    while True:
        line = input('>>> ')
        line = line.rstrip()
        line = line.lstrip()
        command = line.split(' ')[0]
        if len(command) == 0:
            continue
        if command.lower() not in KeyMap.commands.keys() and command.lower() not in KeyMap.commands.values():
            Logger(msg.Errors.invalid_cmd, Logger.warning).log()
            continue

        if len(command) == 1:
            command = list(KeyMap.commands.keys())[list(KeyMap.commands.values()).index(command)]

        fully_exit = parse_n_call(command, line, user)
        if fully_exit is None:
            continue
        if fully_exit:
            exit(0)
        else:
            return


def continue_to_system() -> bool:
    """
    Ask the user to join the system
    :return: True if the user wants to continue to the system
    """
    Logger(msg.Requests.want_to_join, level=Logger.message).log()
    while True:
        ans = input('>>> ')
        if not ans.isalpha():
            continue
        ans = ans.lower()
        if ans == KeyMap.yes:
            return True
        elif ans == KeyMap.no:
            return False
