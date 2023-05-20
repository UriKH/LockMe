"""
Configurations for training and loading the model
"""
import os

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.0006
INPUT_SHAPE = (1, 100, 100)
INPUT_SIZE = (100, 100)
DATASET_PATH = r'C:\LockMe_DATA\my_ATNT_DS'
MODEL_PATH = os.path.join(os.getcwd(), r'model\BCE1_C_BN-1_50_lr0006-vec512.pth')
