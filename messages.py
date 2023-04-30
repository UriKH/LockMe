from keys import KeyMap


class Messages:
    class Errors:
        BUG = ': BUG : sad face...'

        db_no_enc_key = 'Could not retrieve/set an encryption key for dataset. Program will be terminated'
        no_cam = 'No camera available'
        file_exists = f'File does not exists in specified path'

    class Info:
        hello_world = '>>> Hello there! welcome to LockMe service <<<'
        want_to_join = 'Hi there! you are not recognized by the system, would you like to join the system?'
        take_pic = f'to take pic press \'{KeyMap.take_pic}\''
        retake_pic = f'to retake press \'{KeyMap.retake_pic}\' again, else \'{KeyMap.close_cam}\''
        pic_taken = 'picture taken'
        new_db_key = 'congrats! new encryption key for the database was generated'
        file_encrypt = 'file encrypted'
        file_decrypt = 'file decrypted'
