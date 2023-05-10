import cv2 as cv

from camera_runner import Camera
from messages import Messages as msg
from user_login import User
from logger import Logger
import ui
from CNN.initialize import Init


def main():
    try:
        Logger(msg.Info.hello_world, level=Logger.message).log()

        Init()
        cam = Camera()

        while True:
            # activate and run the camera
            cam.run()
            user_img = cam.get_pic()
            user = User(user_img)
            # TODO: add async camera face check: every 10 seconds
            #  (check if the original users face is still in the frame)

            if user.valid:
                Logger('Access Granted', Logger.info).log()
                Logger(msg.Info.user_login + f' {user.uid}', msg.Info).log()

                if ui.present_menu(user):
                    Logger(msg.Info.goodbye, Logger.message).log()
                    exit(0)
            else:
                join = ui.continue_to_system()
                if join:
                    user.uid = Init.database.create_new_user(user.embedding)
                    Logger(msg.Info.user_login + f' {user.uid}', msg.Info).log()

                    if ui.present_menu(user):
                        Logger(msg.Info.goodbye, Logger.message).log()
                        exit(0)
                else:
                    Logger(msg.Info.goodbye, Logger.message).log()
                    exit(0)
    except Exception as e:
        Logger(e, level=Logger.inform).log(main)
    finally:
        del Init.database


if __name__ == '__main__':
    main()
