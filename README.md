# LockMe
 
Welcome to my software engineering class final project.

The project consists of a file encryption app. 
Login is via face detection, thus make sure you have a camera connected to your PC.

# Usage
## Activation
To use the system run in terminal:
1. `pip install -r requirements.txt`
2. and then: `python main.py`

## running example
![img.png](images/terminal_view.png)

# The Model
* I used a classic SNN architecture with 3 convolution blocks and also added batch normalization and dropout.
The network is trained using a cross entropy loss.
I tried also triplet loss and contrastive loss but BCE gave me the best results with the amount of data that I had.
* The model is trained for 50 epochs and learning rate of 0.0006 using batch size 128. While training I used learning rates between 0.0001 to 0.0006.
* The **dataset** is a combination of samples I transformed from AT&T, LFW and my own. 

Here below is the loss of the model after 50 epochs:

![img.png](images/calssic_SNN_with_BN/ATNT+/5%20bce/50%20epochs%20lr0006.png)
