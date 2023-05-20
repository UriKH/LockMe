from messages import Messages as msg
from database import Database as db
from facenet_pytorch import MTCNN
import torch

import model.model as model
from model.train import try_it
from logger import Logger


class Init:
    database = None
    net = None
    mtcnn = None
    device = None

    @Logger(msg.Info.loading, level=Logger.info).time_it
    def __init__(self):
        if not(Init.database is None or Init.net is None or Init.mtcnn is None or Init.device is None):
            return
        Init.database = db()

        Init.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        Init.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=Init.device
        )
        Init.net = model.ClassicModel()
        # model.ClassicModel.get_training_accuracy()
        # try_it()
        state_dict = torch.load(model.config.MODEL_PATH)
        Init.net.load_state_dict(state_dict)
