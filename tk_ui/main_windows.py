import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog
import os
from database import Database
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2

from utils.initialize import Init
from user_login import User
from utils.logger import Logger
from utils.messages import Messages as msg
from tk_ui.ui import Window


class LoginWindow(tk.Frame, Init, Window):
    """
    This class represents the login screen the user sees when opening the application using Tkinter
    """
    image_dims = None

    def __init__(self, root):
        super().__init__(root)
        Init.__init__(self)
        self.root = root

        self.root.resizable(True, True)

        self.frame = tk.Frame(self)
        self.frame.pack()

        self.label = tk.Label(self.frame)
        self.label.pack()

        # Customize button sizes and placement
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=10)

        self.capture_button = ttk.Button(self.button_frame, text="Capture", command=self.capture)
        self.capture_button.pack(side=tk.LEFT, padx=5)

        self.retake_button = ttk.Button(self.button_frame, text="Retake", command=self.retake, state=tk.DISABLED)
        self.retake_button.pack(side=tk.LEFT, padx=5)

        self.login_button = ttk.Button(self, text="Login", command=self.login, state=tk.DISABLED)
        self.login_button.pack(pady=5)

        self.camera = cv2.VideoCapture(0)
        self.captured = False
        self.captured_image = None
        self.user = None

        self.update_camera()  # Start updating the camera view

    def get_window_size(self):
        return self.winfo_width(), self.winfo_height()

    def update_camera(self):
        """
        Update the camera every 10 milliseconds until image capture
        """
        _, frame = self.camera.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.flip(rgb_frame, 1)
        image = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
        self.label.configure(image=image)
        self.label.image = image
        if not self.captured:
            self.after(10, self.update_camera)

    def capture(self):
        """
        Captures a shot using the camera for further face detection processing
        """
        self.captured = True
        self.retake_button.configure(state=tk.NORMAL)
        self.login_button.configure(state=tk.NORMAL)
        self.capture_button.configure(state=tk.DISABLED)

        _, frame = self.camera.read()
        self.captured_image = frame
        self.image_dims = self.captured_image.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.flip(rgb_frame, 1)
        captured_image = Image.fromarray(rgb_frame)
        image = ImageTk.PhotoImage(captured_image)
        self.label.configure(image=image)
        self.label.image = image

    def retake(self):
        """
        Revert to retake state - activate the camera
        """
        self.captured = False
        self.retake_button.configure(state=tk.DISABLED)
        self.login_button.configure(state=tk.DISABLED)
        self.capture_button.configure(state=tk.NORMAL)
        self.update_camera()

    def login(self):
        """
        Try to log in the user. If successful - continues to the application's UI
        """
        if self.captured:
            # Perform face verification using the captured image
            self.user = User(self.captured_image)
            switch_win = False

            if self.user.valid:
                # known person detected
                Logger('Access Granted', Logger.info).log()
                Logger(msg.Info.user_login + f' {self.user.uid}', msg.Info).log()
                switch_win = True
            else:
                # Prompt user to join the system with a popup window
                join = messagebox.askyesno("Login Failed", "Login failed. Would you like to join the system?")
                if join:
                    self.user.uid = self.database.create_new_user(self.user)
                    Logger(msg.Info.user_login + f' {self.user.uid}', msg.Info).log()
                    switch_win = True
                else:
                    self.user = None

            if switch_win:
                image = self.database.get_user_image(self.user.uid, self.image_dims, convert_rgb=True)
                self.show_image_message(image, self.user.uid)
                self.root.switch_frame(MainWindow(self.root, self.get_window_size(), self.user))

    def destroy(self):
        """
        Destroys the current frame and the camera object
        """
        self.camera.release()
        super().destroy()

    def show_image_message(self, image_np, uid):
        # Create a new Toplevel window
        window = tk.Toplevel()

        # Set the "clam" theme for ttk widgets
        style = ttk.Style()
        style.theme_use("clam")

        # Create a PIL image from the NumPy array
        pil_image = Image.fromarray(image_np)

        # Create a Tkinter PhotoImage from the PIL image
        photo = ImageTk.PhotoImage(pil_image)

        # Create a label to display the image
        image_label = ttk.Label(window, image=photo)
        image_label.pack()

        # Create a message label
        message = f"Hi there and welcome user {uid}, score: {self.user.score:.2f}"
        draw = ImageDraw.Draw(pil_image)

        # Set the rectangle parameters
        rect_width = pil_image.width - 20
        rect_height = 50
        rect_position = (10, 10)  # Adjust the position of the rectangle as needed

        # Draw the white rectangle
        draw.rectangle((rect_position, (rect_position[0] + rect_width, rect_position[1] + rect_height)), fill="white")

        # Define the font properties
        font_size = 24
        font = ImageFont.truetype("arial.ttf", font_size)

        # Calculate the text size and position
        text_width, text_height = draw.textsize(message, font=font)
        text_position = ((pil_image.width - text_width) // 2, rect_position[1] + (rect_height - text_height) // 2)

        # Draw the black text on the white rectangle
        draw.text(text_position, message, fill="black", font=font)

        # Convert the updated PIL image to Tkinter PhotoImage
        updated_photo = ImageTk.PhotoImage(pil_image)

        # Update the image label with the updated photo
        image_label.configure(image=updated_photo)
        image_label.image = updated_photo

        # Close the Toplevel window when OK button is clicked
        def close_window():
            window.destroy()

        # Create an OK button
        ok_button = ttk.Button(window, text="OK", command=close_window)
        ok_button.pack()


class MainWindow(tk.Frame, Init, Window):
    """
    This class represents the main screen of the application with which the user could preform various operations
    on files.
    """
    class Buttons:
        """
        Define the buttons of the main window
        """
        add_files = "Add Files"
        rm_files = "Remove Files"
        del_files = "Delete Files"
        show_stat = "Show Status"
        lock_files = "Lock Files"
        unlock_files = "Unlock Files"
        recover_files = "Recover Files"
        logout = "Logout"
        del_user = "Delete User"

        buttons = {}

        @staticmethod
        def buttons_dict(func):
            """
            Create all buttons with the same function
            :param func: the function to activate on button press
            """
            MainWindow.Buttons.buttons.update({
                MainWindow.Buttons.add_files: func,
                MainWindow.Buttons.rm_files: func,
                MainWindow.Buttons.del_files: func,
                MainWindow.Buttons.lock_files: func,
                MainWindow.Buttons.unlock_files: func,
                MainWindow.Buttons.show_stat: func,
                MainWindow.Buttons.recover_files: func
            })

        @staticmethod
        def set_button(button_name, func):
            """
            Set a specific button to a certain function
            :param button_name: the name of the button
            :param func: the function to set the button to
            """
            MainWindow.Buttons.buttons[button_name] = func

    def __init__(self, root, login_window_size, user):
        super().__init__(root)
        Init.__init__(self)
        self.root = root

        self.root.resizable(True, True)
        self.login_window_size = login_window_size
        self.user = user

        self.sidebar_frame = ttk.Frame(self, width=150)
        self.sidebar_frame.pack(fill=tk.Y, side=tk.LEFT)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(padx=10, pady=10)

        # create buttons
        MainWindow.Buttons.buttons_dict(self.create_button)
        MainWindow.Buttons.set_button(MainWindow.Buttons.show_stat, self.create_show_status_frame)
        self.buttons = MainWindow.Buttons.buttons

        self.sidebar_buttons = []
        for button_text in self.buttons.keys():
            button = ttk.Button(self.sidebar_frame, text=button_text,
                                command=lambda text=button_text: self.switch_button_frame(text))
            button.pack(pady=5)
            self.sidebar_buttons.append(button)

        # create other special buttons
        self.logout_button = ttk.Button(self.sidebar_frame, text=MainWindow.Buttons.logout,
                                        style="Yellow.TButton", command=self.logout)
        self.logout_button.pack(pady=10)

        self.delete_user_button = ttk.Button(self.sidebar_frame, text=MainWindow.Buttons.del_user,
                                             style="Red.TButton", command=self.delete_user)
        self.delete_user_button.pack()

        self.current_button_frame = None
        self.table_frame = None

    def create_button(self, button_text):
        """
        General structure for button creation
        :param button_text: the title of the button
        :return: the button's frame
        """
        frame = tk.Frame(self.button_frame)

        add_files_label = tk.Label(frame, text=button_text)
        add_files_label.pack()

        self.choose_files_ui(frame, button_text)
        return frame

    def switch_button_frame(self, button_text):
        """
        switch between the buttons frame when pressing different buttons
        :param button_text: the title of the button
        """
        if self.current_button_frame is not None:
            self.current_button_frame.destroy()

        if button_text == MainWindow.Buttons.show_stat:
            self.current_button_frame = self.create_show_status_frame()
        else:
            create_frame_func = self.buttons.get(button_text)
            if create_frame_func:
                self.current_button_frame = create_frame_func(button_text)

            if self.current_button_frame:
                self.current_button_frame.pack()
                # Remove the status table if it exists
                if self.table_frame is not None:
                    self.table_frame.destroy()
                    self.table_frame = None

    def choose_files_ui(self, frame, command):
        """
        Creating a frane for selecting files
        :param frame: the button's frame
        :param command: the name of the command the button preforms
        """
        file_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, width=50, height=10)
        file_listbox.pack()

        def choose_files():
            nonlocal command

            selected_paths = filedialog.askopenfilenames()
            if selected_paths:
                for file_path in selected_paths:
                    file_listbox.insert(tk.END, file_path)
                self.run_command(selected_paths, command)

        choose_files_button = ttk.Button(frame, text="Choose Files", command=choose_files)
        choose_files_button.pack()

    def run_command(self, paths, cmd):
        """
        Run the specified command function
        :param paths: paths of files to preform the operation on
        :param cmd: the command to run
        """

        error_paths = []
        for path in paths:
            result = True
            try:
                match cmd:
                    case MainWindow.Buttons.add_files:
                        result = self.database.add_file(path, self.user)
                    case MainWindow.Buttons.rm_files:
                        result = self.database.remove_file(path, self.user)
                    case MainWindow.Buttons.lock_files:
                        result = self.database.lock_file(path, self.user)
                    case MainWindow.Buttons.unlock_files:
                        result = self.database.unlock_file(path, self.user)
                    case MainWindow.Buttons.del_files:
                        result = self.database.remove_file(path, self.user)
                        if result:
                            os.remove(path)
                            Logger(msg.Info.file_deleted + f' {path}', Logger.warning).log()
                    case MainWindow.Buttons.recover_files:
                        result = self.database.recover_file(path, self.user)
                if not result:
                    error_paths.append(path)
            except Exception as e:
                Logger(e, Logger.inform).log()
                Logger(msg.Info.back_to_routine, Logger.info).log()

        if len(error_paths) != 0:
            err_msg = f'Operation {cmd} failed on files:\n'
            for path in error_paths:
                err_msg += path + '\n'
            messagebox.showerror('Operation failed', err_msg + 'try again or view manual for help')

    @staticmethod
    def clean_data(data: dict):
        """
        Present the data of the user
        :param data: a dictionary representing the user's data
        :return: reformat the data: {file-path: (file suffix, file state)}
        """
        data = data.copy()

        del data['file']
        del data['user_id']
        data['file_state'] = ['locked' if state == Database.file_state_locked else 'open' for state in
                              data['file_state']]
        return {data['file_path'][i]: (data['suffix'][i], data['file_state'][i], ) for i in range(len(data['file_path']))}

    def create_show_status_frame(self, *args):
        """
        Create the user's files status table
        """
        if self.table_frame is not None:
            return self.table_frame

        frame = tk.Frame(self.button_frame)

        show_status_label = tk.Label(frame, text="Show Status")
        show_status_label.pack()

        data = self.database.fetch_user_data(self.user.uid)
        data = MainWindow.clean_data(data)

        table = ttk.Treeview(frame, columns=("File path", "File suffix", "State"))
        table.heading("File path", text="File path")
        table.heading("File suffix", text="File suffix")
        table.heading("State", text="State")

        table.column("#0", width=0)

        for file, val in data.items():
            table.insert("", tk.END, values=(file, *val))

        table.pack(fill=tk.BOTH, expand=True)

        self.table_frame = frame
        self.table_frame.pack()  # Pack the frame into the parent frame

        return frame

    def logout(self):
        """
        Log out the current user
        """
        Logger(msg.Info.logging_off, Logger.message).log()
        self.database.lock_all_files(self.user.uid)
        self.root.switch_frame(LoginWindow(self.root))

    def delete_user(self):
        """
        Delete the current user from the system
        """
        self.database.delete_user(self.user.uid, sure=True)
        Logger(msg.Info.logging_off, Logger.message).log()
        self.root.switch_frame(LoginWindow(self.root))

    def destroy(self):
        """
        Destroy the window
        """
        if self.table_frame is not None:
            self.table_frame.destroy()
        super().destroy()

    def get_window_size(self):
        return self.winfo_width(), self.winfo_height()
