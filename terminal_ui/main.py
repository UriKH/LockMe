from camera_runner import Camera
from user_login import User
from utils.logger import Logger
from utils.initialize import Init
from utils.messages import Messages as msg

from . import ui


def main():
    try:
        Logger(msg.Info.hello_world, level=Logger.message).log(msg_prefix=' ')

        Init()
        Logger('\n' + msg.Info.barrier).log(msg_prefix=' ')
        cam = Camera()

        while True:
            # activate and run the camera
            cam.run()
            Logger(msg.Info.barrier + '\n').log(msg_prefix='')
            user_img = cam.get_pic()
            user = User(user_img)

            if user.valid:
                # known person detected
                Logger('Access Granted', Logger.info).log()
                Logger(msg.Info.user_login + f' {user.uid}', msg.Info).log()

                ui.present_menu(user)
            else:
                # unrecognized person detected
                join = ui.continue_to_system()
                if join:
                    user.uid = Init.database.create_new_user(user.embedding)
                    Logger(msg.Info.user_login + f' {user.uid}', msg.Info).log()
                    ui.present_menu(user)
                else:
                    exit(0)
    except Exception as e:
        Logger(e, level=Logger.inform).log(main)
    finally:
        del Init.database


if __name__ == '__main__':
    main()
