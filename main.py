from camera_runner import Camera
from messages import Messages as msg
import login
from logger import Logger
import ui


def main():
    Logger(msg.Info.hello_world, level=Logger.message).log()

    # activate and run the camera
    cam = Camera()
    cam.run()
    user_img = cam.get_pic()

    # check user validation
    # TODO: here CNN
    # encoding = None
    # valid, uid = login.login(encoding)
    #
    # if valid:
    #     data = data.retrieve(uid)
    #     ui.present_menu()
    #     ui.present_data(data)
    # else:
    #     Logger(msg.WantToJoin, level=Logger.message).log()


if __name__ == '__main__':
    main()
