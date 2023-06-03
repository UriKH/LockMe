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

# DATASET_PATH = r''#r'<PATH TO YOUR DATA SET TO TRAIN ON>'
TRAIN_DATASET_PATH = r'C:\LockMe_DATA\my_ATNT_DS\TRAIN'
TEST_DATASET_PATH = r'C:\LockMe_DATA\my_ATNT_DS\TEST'

MODEL_NAME = r'BCE2_C_BN-1_30_lr0006.pth'
MODEL_PATH = os.path.join(os.getcwd(), MODEL_NAME)

OUT_NAME = r'model.pth'
OUT_PATH = os.path.join(os.getcwd(), OUT_NAME)

