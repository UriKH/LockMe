"""
Key mapping for the project's UI
"""


class KeyMap:
    take_pic = 't'
    close_cam = 'c'
    retake_pic = 'r'
    yes = 'y'
    no = 'n'
    add = 'a'
    remove = 'r'
    delete = 'd'
    trash = 't'
    log_off = 'o'
    exit = 'x'
    show = 's'
    lock = 'l'
    unlock = 'u'
    exit_cmd = 'exit'
    add_cmd = 'add'
    remove_cmd = 'remove'
    delete_cmd = 'delete'
    trash_cmd = 'trash'
    log_off_cmd = 'logoff'
    show_cmd = 'show'
    lock_cmd = 'lock'
    lock_all_cmd = 'lock_all'
    unlock_cmd = 'unlock'
    unlock_all_cmd = 'unlock_all'
    recover_cmd = 'recover'

    commands = {exit_cmd: exit, log_off_cmd: log_off,
                add_cmd: add, remove_cmd: remove,
                delete_cmd: delete, trash_cmd: trash,
                show_cmd: show,
                lock_cmd: lock, lock_all_cmd: lock_all_cmd,
                unlock_cmd: unlock, unlock_all_cmd: unlock_all_cmd,
                recover_cmd: recover_cmd}

    cmd_param = {add_cmd: 1, remove_cmd: 1,
                 delete_cmd: 0, trash_cmd: 1,
                 log_off_cmd: 0, exit_cmd: 0,
                 show_cmd: 0,
                 lock_cmd: 1, lock_all_cmd: 0,
                 unlock_cmd: 1, unlock_all_cmd: 0,
                 recover_cmd: 1}
