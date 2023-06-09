"""
Configurations for training and loading the model
"""
import os

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0006
INPUT_SHAPE = (1, 105, 105)
INPUT_SIZE = (105, 105)
FINE_TUNE = True

TRAIN_DATASET_PATH = r'C:\LockMe_DATA\my_ATNT_DS\TRAIN'
TEST_DATASET_PATH = r'C:\LockMe_DATA\my_ATNT_DS\TEST'

MODEL_NAME = r'current-testit.pth'
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_NAME)

OUT_NAME = r'model.pth'
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME)

