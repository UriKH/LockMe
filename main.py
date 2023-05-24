from camera_runner import Camera
from messages import Messages as msg
from user_login import User
from logger import Logger
import ui
from initialize import Init


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
                Logger('Access Granted', Logger.info).log()
                Logger(msg.Info.user_login + f' {user.uid}', msg.Info).log()

                ui.present_menu(user)
            else:
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
