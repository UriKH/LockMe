from keys import KeyMap


class Messages:
    class Errors:
        BUG = ': BUG : sad face...'

        db_no_enc_key = 'Could not retrieve/set an encryption key for dataset. Program will be terminated'
        no_cam = 'No camera available'
        file_exists = f'File does not exists in specified path'
        no_face_click = 'face was not clicked'

    class Info:
        hello_world = '>>> Hello there! welcome to LockMe service <<<'
        take_pic = f'To take pic press \'{KeyMap.take_pic}\''
        retake_pic = f'To retake press \'{KeyMap.retake_pic}\' again, else \'{KeyMap.close_cam}\''
        pic_taken = 'Picture taken'
        new_db_key = 'Congrats! new encryption key for the database was generated'
        file_encrypt = 'File encrypted'
        file_decrypt = 'File decrypted'
        embeddings_generated = 'Face embeddings generated'
        faces_located = 'Faces located'
        loading = 'Loading and initiating the app'
        goodbye = 'Goodbye and thank you for using LockMe'

    class Requests:
        want_to_join = 'Hi there! you are not recognized by the system, would you like to join the system? [Y/n]'
        face_index = 'Please click on the user\'s face'
