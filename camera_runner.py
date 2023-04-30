import cv2 as cv
from time import sleep
from keys import KeyMap
from messages import Messages as msg
from messages import Errors as err
from Logger import Logger


class Camera:
    default_size = 500
    window_name = 'cam view'
    freeze_color = (0, 255, 0)
    retake_time = 3

    def __init__(self):
        self.v_cap = cv.VideoCapture(0)
        if self.v_cap is None:
            err.error(msg.NoCam)

        self.pic = None
        self.last_frame = None

    def run(self):
        Logger.log_instruction(msg.TakePic)
        taken = False

        while True:
            self.read_stream()
            image = self.prepare_presentation()
            cv.imshow(Camera.window_name, image)

            # retake the image
            key = cv.waitKey(1)
            if taken and key == ord(KeyMap.take_pic):
                cv.imshow(Camera.window_name, self.freeze())
                cv.waitKey(Camera.retake_time * 1000)
                cv.destroyAllWindows()
                return

            # taking the image
            elif key == ord(KeyMap.take_pic):
                cv.imshow(Camera.window_name, self.freeze())
                Logger.log_instruction(msg.RetakePic)
                taken = True

                while True:
                    cv.imshow(Camera.window_name, self.freeze())
                    key = cv.waitKey(1)

                    if key == ord(KeyMap.close_cam):
                        Logger.log_instruction(msg.PicTaken)
                        return
                    elif key == ord(KeyMap.take_pic):
                        Logger.log_instruction(msg.TakePic)
                        break


    def read_stream(self):
        ret, frame = self.v_cap.read()
        self.pic = frame if frame is not None else self.last_frame
        self.last_frame = self.pic

    def prepare_presentation(self):
        image = self.pic.copy()
        h, w, _ = image.shape
        cv.resize(image, (int(Camera.default_size * w/h), Camera.default_size))
        return image

    def freeze(self):
        image = self.prepare_presentation()
        cv.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), Camera.freeze_color, 2)
        return image