"""
Configurations for training and loading the model
"""
import os

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.0006
INPUT_SHAPE = (1, 105, 105)
INPUT_SIZE = (105, 105)
DATASET_PATH = r'<PATH TO YOUR DATA SET TO TRAIN ON>'
MODEL_NAME = r"BCE2_C_BN-1_30_lr0006.pth"#r'model.pth'
MODEL_PATH = os.path.join(os.getcwd(), MODEL_NAME)
FINE_TUNE = False
OUT_NAME = r'model.pth'
OUT_PATH = os.path.join(os.getcwd(), OUT_NAME)

