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

        # activate and run the camera
        cam = Camera()
        cam.run()
        user_img = cam.get_pic()
        user = User(user_img)

        if user.valid:
            Logger('Access Granted', Logger.info).log()
            # user_data = Init.database.retrive_user_data()
            # ui.present_menu()
            # ui.present_data(data)
        else:
            join = ui.continue_to_system()
            if join:
                uid = Init.database.create_new_user(user.embedding)
            else:
                Logger(msg.Info.goodbye, Logger.message).log()
                exit(0)
    except Exception as e:
        Logger(e, level=Logger.inform).log()
    finally:
        del Init.database


if __name__ == '__main__':
    main()
