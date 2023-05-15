from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import config
import cv2 as cv
from tqdm import tqdm


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    if not os.path.exists(config.DATA_PATH):
        print('Path does not exists')
        return

    names = os.listdir(config.DATA_PATH)
    for name in tqdm(names):
        person_path = os.path.join(config.DATA_PATH, name)
        if not os.path.isdir(person_path):
            continue

        image_paths = [os.path.join(person_path, image_name) for image_name in os.listdir(person_path)
                       if os.path.isfile(os.path.join(person_path, image_name))]
        person_out = os.path.join(config.FACES_OUT_DIR, name)
        if not os.path.exists(person_out):
            os.makedirs(person_out)

        for image_path in image_paths:
            image_id = image_path.split(os.path.sep)[-1]
            image = cv.imread(image_path)
            h, w, _ = image.shape

            # now do face detection and stuff
            boxes, conf = mtcnn.detect(image)
            boxes = boxes.astype(int)
            boxes = [box for i, box in enumerate(boxes) if conf[i] >= 0.95]


if __name__ == '__main__':
    main()
