import keras_facenet as kfn
from logger import Logger
from messages import Messages as msg
from database import Database as db


class Init:
    facenet_model = None
    database = None

    @Logger(msg.Info.loading, level=Logger.info).time_it
    def __init__(self):
        Init.facenet_model = kfn.FaceNet()
        Init.database = db()
