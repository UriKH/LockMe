# LockMe
 
Welcome to my software engineering class final project.

The project consists of a file encryption app. 
Login is via face detection, thus make sure you have a camera connected to your PC.

# Usage
## Activation
To activate the system follow these steps:
1. run: `pip install -r requirements.txt`
2. use a Unix-based system to rebuild the model from parts in the **_model/model_parts_** directory using the command: 
`cat model.part* > model.pth`
3. make sure the model name is written with the correct path in the configurations file in the model's directory
4. you are then ready to run: `python main.py`

## running example
![img.png](images/terminal_view.png)

# The Model
* I used the classic SNN architecture using binary cross entropy loss.
* I tried a few models and loss functions such as triplet loss and contrastive loss with different learning rates 
but BCE gave me the best results with the amount of data that I had.
* The model is trained for 50 epochs and learning rate of 0.0006 using batch size 128.
* The **dataset** is a combination of samples I transformed from AT&T, LFW and my own. 

Here below is the loss and accuracy of the model after 50 epochs:

![img.png](images/img.png)
