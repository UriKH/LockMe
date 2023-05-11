import os
import random
import numpy as np


class TripletGenerator:
    def __init__(self, ds_path):
        self.people_paths = []

        for directory in os.listdir(ds_path):
            directory_path = os.path.join(ds_path, directory)
            amount = len(os.listdir(directory_path))

            if amount > 1:
                self.people_paths.append(directory_path)

        self.all_people_dict = self.generate_people_dict()

    def generate_people_dict(self):
        people = {}
        for person_path in self.people_paths:
            images_names = os.listdir(person_path)
            people[person_path] = [os.path.join(person_path, image_name) for image_name in images_names]
        return people

    def get_next_triplet(self):
        while True:
            anchor_person = random.choice(self.people_paths)

            temp = self.people_paths.copy()
            temp.remove(anchor_person)
            negative_person = random.choice(temp)
            negative = random.choice(self.all_people_dict[negative_person])

            anchor, positive = np.random.choice(
                a=self.all_people_dict[anchor_person],
                size=2,
                replace=False
            )

            yield anchor, positive, negative
