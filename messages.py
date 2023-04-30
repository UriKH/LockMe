from keys import KeyMap


class Messages:
    NoCam = 'No camera available'
    TakePic = f'to take pic press \'{KeyMap.take_pic}\''
    RetakePic = f'to retake press \'{KeyMap.take_pic}\' again, else \'{KeyMap.close_cam}\''
    PicTaken = 'picture taken'


class Errors:
    @staticmethod
    def error(msg):
        raise Exception(msg)

    @staticmethod
    def terminate():
        print('fatal error - exiting')
        exit(1)
