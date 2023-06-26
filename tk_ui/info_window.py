
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkfont
import os
from PIL import Image, ImageTk

from tk_ui.ui import Window
from tk_ui.main_windows import LoginWindow


goals_msg = """ The goal of this project is to help solving the security problem of files in shared computers.
 The future app is planed to include cloud management of the users, folder encryption, facial recognition 
 liveness check and advanced user file recovery methods.
"""
description_msg = """ This is the prototype of the LockMe application. 
 LockMe is a python application in development stages for individuals who share their computer with others
 and want to keep their files safe and secure. 
 LockMe provides encryption service based on facial recognition and CBC (Cypher Block Chaining) using 
 AES (Advanced Encryption Standard) encryption algorithm.

 The application consists the following features:
  1. Adding unlimited files to the system - the files are backed-up and saved separately 
  2. Encrypting files
  3. Decrypting files
  4. Removing files
  5. Recovering files from their last updated version
  6. Viewing file status
  7. Using command line interface
"""

model_msg = """ The model architecture used in this project is the siamese architecture as presented in the paper. 
 The model architecture can be seen in the following scheme: 
 """

training_msg_1 = """ The model is trained using BCE (Binary Cross Entropy) loss function. 
 The model was trained using learning rate 0.0006 and batch size 128 for 50 epochs using the Adam 
 optimizer. Also the batch normalization method was used. 
 The model was trained on the combined dataset of LFW, AT&T and some images taken by Uri K.H.
 The train-test ratio of the dataset after augmentations is 92%-8% (total of 33,275 images).
 Some example images from the dataset after augmentation (0 - same subject, 1 - different subjects):
 """

training_msg_2 = """ The model has 98% accuracy on the train set and just over 80% on the test set. 
 More sophisticated models are still in testing stages. Now in testings are being conducted on the 
 VGG19/16 and Inception-resnet-v2 in Siamese NN and in Triplet Loss usage. 
 The testing datasets are LFW and CelebA (testing code notebook could be found on
 github: https://github.com/UriKH/LockMe_Face-Recognition). 
 
 In the example below are some pairs of the augmented CelebA dataset:"""


class InfoWindow(tk.Frame, Window):
    image_resize_factor = {'small': 0.6, 'medium': 0.8, 'big': 0.95}

    def __init__(self, root):
        super().__init__(root)
        self.root = root

        # Configure row and column weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create a text widget with a vertical scrollbar
        self.text = tk.Text(self, width=50, height=10, state="disabled")

        # Initiate scrollbar
        self.scrollbar = ttk.Scrollbar(self, command=self.text.yview)
        self.scrollbar.grid(row=0, column=1, sticky=tk.NS)

        self.generate_text()

        # Create a button below the text widget
        self.button = ttk.Button(self, text='Login/Signup', command=self.exit_intro)
        self.button.grid(row=1, column=0, pady=10, sticky=tk.S)

        # Configure scrollbar fill
        self.scrollbar.grid(sticky=tk.NS + tk.E + tk.W)

    def get_window_size(self):
        return self.winfo_width(), self.winfo_height()

    def exit_intro(self):
        """
        Switch to the login window
        """
        self.root.switch_frame(LoginWindow(self.root))

    def generate_text(self):
        """
        Generate the relevant text in the window
        """
        self.text.configure(yscrollcommand=self.scrollbar.set)
        self.text.grid(row=0, column=0, sticky=tk.NSEW)

        font = tkfont.Font(family='Tahoma', size=16, weight="bold", underline=True)
        self.text.tag_configure("headline1", font=font)
        font = tkfont.Font(family='Tahoma', size=14, weight="bold")
        self.text.tag_configure("headline2", font=font)
        font = tkfont.Font(family='Tahoma', size=13, weight="bold")
        self.text.tag_configure("headline3", font=font)
        font = tkfont.Font(family='Trebuchet MS', size=12)
        self.text.tag_configure("text", font=font)

        self.text.configure(state="normal")  # Enable editing temporarily

        self.text.insert(tk.END, ' ', 'text')
        self.text.insert(tk.END, 'Wellcome to LockMe service\n', 'headline1')
        self.text.insert(tk.END, ' Goals\n', 'headline2')
        # goals of the project
        self.text.insert(tk.END, goals_msg + '\n', 'text')
        self.text.insert(tk.END, ' Description\n', 'headline2')
        # description of the project
        self.text.insert(tk.END, description_msg + '\n', 'text')
        self.text.insert(tk.END, ' About\n', 'headline2')
        self.text.insert(tk.END, ' The model\n', 'headline3')
        # model description part
        self.text.insert(tk.END, model_msg + '\n', 'text')
        self.add_image(os.path.join(os.getcwd(), 'images', 'siamese_model.png'), 'big')
        self.text.insert(tk.END, '\n Training\n', 'headline3')
        # training part
        self.text.insert(tk.END, training_msg_1 + '\n', 'text')
        self.add_image(os.path.join(os.getcwd(), 'images', 'example1.png'), 'small')
        self.text.insert(tk.END, ' [0    1    0    0    1    0    0    1]\n\n', 'text')
        self.add_image(os.path.join(os.getcwd(), 'images', 'example2.png'), 'small')
        self.text.insert(tk.END, ' [0    0    0    1    0    0    1    1]\n\n', 'text')
        self.text.insert(tk.END, training_msg_2 + '\n', 'text')
        self.add_image(os.path.join(os.getcwd(), 'images', 'celeb_a_transformed.png'), 'small')
        self.text.insert(tk.END, ' [0    1    0    1    0    0    0    1]\n\n', 'text')

        self.text.configure(state="disabled")  # Disable editing again

    def add_image(self, image_path, size_factor: str):
        """
        Add image to the text view
        :param image_path: the path to the image to be added
        :param size_factor: the multiplier to fit the image to the window by
        """
        # Load the image using PhotoImage
        image = Image.open(image_path)

        # resize the image
        width, height = image.size
        prop = height / width
        win_width = round(self.root.get_window_size()[0] * self.image_resize_factor[size_factor])
        image = image.resize((win_width, round(prop * win_width)))  # Resize the image if needed

        photo = ImageTk.PhotoImage(image)

        # Create a label widget to display the image
        label = tk.Label(self, image=photo)
        label.image = photo  # Keep a reference to prevent garbage collection

        # Insert the label widget as a window within the text widget
        self.text.window_create(tk.END, window=label)
        self.text.insert(tk.END, '\n')  # Add a newline after the image

        def on_mousewheel(event):
            self.text.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Disable hovering effect and mouse wheel binding on the label widget
        label.bind_all("<MouseWheel>", on_mousewheel)


if __name__ == '__main__':
    pass
