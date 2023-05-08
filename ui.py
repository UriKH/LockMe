# all other classes like camera_runner should inherit from this class.
# this class will define the outputs which are not from the terminal
import cv2 as cv
from logger import Logger
from messages import Messages as msg
from keys import KeyMap


def present_menu():
    Logger(msg.Info.menu, Logger.info).log()
    while True:
        # TODO: parse user input and activate commands
        exit(0)


def present_data(data):
    # TODO: write a function for data representation (maybe use pandas for table representation?)
    pass


def continue_to_system():
    Logger(msg.Requests.want_to_join, level=Logger.message).log()
    while True:
        key = cv.waitKey()
        if key == ord(KeyMap.yes):
            return True
        elif key == ord(KeyMap.no):
            return False
