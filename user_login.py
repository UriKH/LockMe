import torch
import numpy as np

from image_process import Image
from initialize import Init
from database import Database
from logger import Logger
from messages import Messages as msg


class User:
    dist_thresh = 0.5

    def __init__(self, user_img):
        self.img_data = Image(user_img)
        self.embedding = self.img_data.choose_face()
        self.uid = None
        self.valid = self.login()

    def check_similarity(self, data):
        """
        Get the user ID
        :param data: data as [(ID, embedding) ...]
        :return: the ID of the user if in the DB else None
        """
        min_dist = 100
        identity = None

        dist = 0
        for uid, e in data:
            e = torch.from_numpy(np.array(list(e))).to(torch.float32)
            dist = Init.net.forward_embeddings(e, self.embedding).item()
            if dist < min_dist:
                min_dist = dist
                identity = uid
        if min_dist > User.dist_thresh:
            identity = None
        Logger(msg.Info.login_distance_func + f' {dist:.2f}', Logger.info).log()
        return identity

    def login(self):
        """
        Try to log in with the current user
        :return: if the user is in the system or not
        """
        data = Init.database.fetch_users()
        for i, user in enumerate(data):
            data[i] = list(data[i])
            data[i][1] = Database._byte_to_embedding(user[1])
        self.uid = self.check_similarity(data)
        return self.uid is not None
