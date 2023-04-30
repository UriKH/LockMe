import cv2 as cv
from keys import KeyMap
from messages import Messages as msg
from logger import Logger


class Camera:
    default_size = 500
    window_name = 'cam view'
    freeze_color = (0, 255, 0)
    retake_time = 3

    def __init__(self):
        """
        Initialize the Camera for scanning
        """
        self._v_cap = cv.VideoCapture(0)
        if self._v_cap is None:
            Logger(msg.Errors.NoCam, level=Logger.warning).log()
            # TODO: pass to user and password login

        self._pic = None
        self._last_frame = None

    def run(self):
        """
        Run the camera UI to capture an image of the user
        """
        Logger(msg.TakePic, level=Logger.message).log()
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
                Logger(msg.Info.RetakePic, level=Logger.message).log()
                taken = True

                while True:
                    cv.imshow(Camera.window_name, self.freeze())
                    key = cv.waitKey(1)

                    if key == ord(KeyMap.close_cam):
                        Logger(msg.Info.PicTaken, level=Logger.message).log()
                        return
                    elif key == ord(KeyMap.take_pic):
                        Logger(msg.Info.TakePic, level=Logger.message).log()
                        break

    def read_stream(self):
        """
        Read a frame from the camera
        """
        ret, frame = self._v_cap.read()
        self._pic = frame if frame is not None else self._last_frame
        self._last_frame = self._pic

    def prepare_presentation(self):
        """
        Resize image for later presentation
        """
        image = self._pic.copy()
        h, w, _ = image.shape
        cv.resize(image, (int(Camera.default_size * w/h), Camera.default_size))
        return image

    def freeze(self):
        """
        Create a frozen frame representation
        """
        image = self.prepare_presentation()
        cv.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), Camera.freeze_color, 2)
        return image

    def get_pic(self):
        return self._pic.copy()