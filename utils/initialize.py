from facenet_pytorch import MTCNN
import torch

from utils.messages import Messages as msg
from database import Database as db
from model.SNN import Net
from model import config
from utils.logger import Logger


class Init:
    """
    This class represents all the objects the application needs to initialize and use
    """
    database = None
    net = None
    mtcnn = None
    device = None

    @Logger(msg.Info.loading, level=Logger.info).time_it
    def __init__(self):
        if Init.database is None:
            # initiate the database object
            Init.database = db()

        if Init.mtcnn is None:
            # initiate the MTCNN network for later face detection
            Init.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            Init.mtcnn = MTCNN(
                image_size=160, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                device=Init.device
            )

        if Init.net is None:
            # load the model to be ready for use for face recognition
            Init.net = Net()
            state_dict = torch.load(config.MODEL_PATH)
            Init.net.load_state_dict(state_dict)

