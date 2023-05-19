from keys import KeyMap
from logger import Logger


class Messages:
    class Errors:
        """
        This class represents all the constant error messages
        """
        BUG = 'BUG sad face...'

        db_no_enc_key = 'Could not retrieve/set an encryption key for dataset. Program will be terminated'
        no_cam = 'No camera available'
        file_exists = f'File does not exists in specified path'
        no_face_click = 'Face was not clicked'
        invalid_cmd = 'Command is invalid'
        missing_param = 'Missing parameter'
        to_many_params = 'To many parameters for this command'
        failed_insertion = 'Failed to add the file, file already has an owner'
        access_denied = 'Access denied'
        failed_removal = 'File not in the system'
        unsupported_file_type = 'Unsupported file type'

    class Info:
        """
        This class represents all the constant general messages to the user
        """
        hello_world = '>>> Hello there! welcome to LockMe service <<<'
        take_pic = f'To take a picture press \'{KeyMap.take_pic}\' (IMPORTANT: look at the camera)'
        retake_pic = f'To retake the image press \'{KeyMap.retake_pic}\' again, else \'{KeyMap.close_cam}\' to close'
        pic_taken = 'Picture taken'
        new_db_key = 'Congrats! new encryption key for the database was generated'
        file_encrypt = 'File encrypted'
        file_decrypt = 'File decrypted'
        file_added = 'File added to the system'
        file_removed = 'File removed from the system'
        file_deleted = 'File deleted from the disk'
        file_recovered = 'File recovered successfully - the file was corrupted'
        embeddings_generated = 'Face embeddings generated'
        faces_located = 'Faces located'
        loading = 'Loading and initiating the app'
        goodbye = 'Goodbye and thank you for using LockMe\n'
        exiting = 'Exiting'
        logging_off = 'logging-off the current user'
        single_face = 'Only one face detected, using this face by default'
        user_login = 'User logged in with ID'
        user_deleted = 'User removed from the system'

        menu = f"""Available commands:
 > {KeyMap.add_cmd}      [{KeyMap.add}] -> add a new file to the system
 > {KeyMap.remove_cmd}   [{KeyMap.remove}] -> remove a file from the system
 > {KeyMap.trash_cmd}    [{KeyMap.trash}] -> delete a file from disk
 > {KeyMap.show_cmd}     [{KeyMap.show}] -> list all files of the user and their status
 > {KeyMap.lock_cmd}     [{KeyMap.lock}] -> lock a file with encryption
 > {KeyMap.lock_all_cmd}     -> lock all files
 > {KeyMap.unlock_all_cmd}   -> unlock all files
 > {KeyMap.unlock_cmd}   [{KeyMap.unlock}] -> unlock a file
 > {KeyMap.recover_cmd}      -> recover a file
 > {KeyMap.delete_cmd}   [{KeyMap.delete}] -> delete the user account and unlock all of its files
 > {KeyMap.log_off_cmd}   [{KeyMap.log_off}] -> log off the system
 > {KeyMap.exit_cmd}     [{KeyMap.exit}] -> exit the program"""
        back_to_routine = 'Error ignored: Back to routine'
        barrier = '------------------------------------------------------------------------'

    class Requests:
        """
        This class represents all the constant requests messages
        """
        want_to_join = 'Hi there! you are not recognized by the system, would you like to join the system? [Y/n]'
        face_index = 'Please click on the user\'s face'
        delete_user = 'Are you sure you want to delete all user data? [y/N]'

    class Load:
        unlocking_files = f'{Logger.Colors.yellow}Unlock files'
        locking_files = f'{Logger.Colors.yellow}Lock files '
