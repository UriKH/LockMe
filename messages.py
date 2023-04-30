from keys import KeyMap


class Messages:
    class Errors:
        BUG = ': BUG : sad face...'

        DB_NoEncKey = 'Could not retrieve/set an encryption key for dataset. Program will be terminated'
        NoCam = 'No camera available'

    class Info:
        HelloWorld = '>>> Hello there! welcome to LockMe service <<<'
        WantToJoin = 'Hi there! you are not recognized by the system, would you like to join the system?'
        TakePic = f'to take pic press \'{KeyMap.take_pic}\''
        RetakePic = f'to retake press \'{KeyMap.retake_pic}\' again, else \'{KeyMap.close_cam}\''
        PicTaken = 'picture taken'
