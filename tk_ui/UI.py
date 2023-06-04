import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import cv2
from PIL import Image, ImageTk
import os

from messages import Messages as msg
from logger import Logger
from initialize import Init
from user_login import User
from keys import KeyMap
from database import Database


class BaseWindow(tk.Tk):
    def __init__(self, title):
        super().__init__()
        self.title(title)

        # Apply a theme
        style = ttk.Style()
        self.resizable(True, True)
        style.theme_use("clam")  # You can choose a different theme if desired
        style.configure("Red.TButton", background="red")
        style.configure("Yellow.TButton", background="yellow")
        self.current_frame = None

    def switch_frame(self, new_frame):
        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = new_frame

        if isinstance(new_frame, LoginWindow):
            self.current_frame.pack()
        elif isinstance(new_frame, MainWindow):
            self.current_frame.pack(fill=tk.BOTH, expand=True)
            self.geometry(f"{new_frame.login_window_size[0]}x{new_frame.login_window_size[1]}")


class LoginWindow(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

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
        _, frame = self.camera.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
        self.label.configure(image=image)
        self.label.image = image
        if not self.captured:  # Only update when not captured
            self.after(10, self.update_camera)  # Update every 10 milliseconds

    def capture(self):
        self.captured = True
        self.retake_button.configure(state=tk.NORMAL)  # Enable retake button
        self.login_button.configure(state=tk.NORMAL)  # Enable login button
        self.capture_button.configure(state=tk.DISABLED)  # Disable capture button

        _, frame = self.camera.read()
        self.captured_image = frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        captured_image = Image.fromarray(rgb_frame)
        image = ImageTk.PhotoImage(captured_image)
        self.label.configure(image=image)
        self.label.image = image

    def retake(self):
        self.captured = False
        self.retake_button.configure(state=tk.DISABLED)  # Disable retake button
        self.login_button.configure(state=tk.DISABLED)  # Disable login button
        self.capture_button.configure(state=tk.NORMAL)  # Enable capture button
        self.update_camera()  # Resume updating the camera view

    def login(self):
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
                join = messagebox.askyesno("Login Failed", "Login failed. Do you want to join the system?")
                if join:
                    self.user.uid = Init.database.create_new_user(self.user.embedding)
                    Logger(msg.Info.user_login + f' {self.user.uid}', msg.Info).log()
                    switch_win = True
                else:
                    self.user = None

            if switch_win:
                self.parent.switch_frame(MainWindow(self.parent, self.get_window_size(), self.user))

    def clear_content(self):
        self.frame.pack_forget()
        self.button_frame.pack_forget()
        self.login_button.pack_forget()
        self.label.pack_forget()

    def destroy(self):
        self.camera.release()
        super().destroy()


class MainWindow(tk.Frame):
    def __init__(self, parent, login_window_size, user):
        super().__init__(parent)
        self.parent = parent
        self.login_window_size = login_window_size
        self.user = user

        self.sidebar_frame = ttk.Frame(self, width=150)
        self.sidebar_frame.pack(fill=tk.Y, side=tk.LEFT)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(padx=10, pady=10)

        self.buttons = {
            KeyMap.add_files: self.create_button,
            KeyMap.rm_files: self.create_button,
            KeyMap.del_files: self.create_button,
            KeyMap.lock_files: self.create_button,
            KeyMap.unlock_files: self.create_button,
            KeyMap.show_stat: self.create_show_status_frame,
            KeyMap.recover_files: self.create_button
        }

        self.sidebar_buttons = []
        for button_text in self.buttons.keys():
            button = ttk.Button(self.sidebar_frame, text=button_text, command=lambda text=button_text: self.switch_button_frame(text))
            button.pack(pady=5)
            self.sidebar_buttons.append(button)

        self.logout_button = ttk.Button(self.sidebar_frame, text=KeyMap.logout, style="Yellow.TButton", command=self.logout)
        self.logout_button.pack(pady=10)

        self.delete_user_button = ttk.Button(self.sidebar_frame, text=KeyMap.del_user, style="Red.TButton", command=self.delete_user)
        self.delete_user_button.pack()

        self.current_button_frame = None

        self.table_frame = None

    def create_button(self, button_text):
        frame = tk.Frame(self.button_frame)

        add_files_label = tk.Label(frame, text=button_text)
        add_files_label.pack()

        self.choose_files_ui(frame, button_text)
        return frame

    def switch_button_frame(self, button_text):
        if self.current_button_frame is not None:
            self.current_button_frame.destroy()

        if button_text == KeyMap.show_stat:
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
        for path in paths:
            try:
                if cmd == KeyMap.add_files:
                    Init.database.add_file(path, self.user)
                elif cmd == KeyMap.rm_files:
                    Init.database.remove_file(path, self.user)
                elif cmd == KeyMap.lock_files:
                    Init.database.lock_file(path, self.user)
                elif cmd == KeyMap.unlock_files:
                    Init.database.unlock_file(path, self.user)
                elif cmd == KeyMap.del_files:
                    Init.database.remove_file(path, self.user)
                    os.remove(path)
                    Logger(msg.Info.file_deleted + f' {path}', Logger.warning).log()
                elif cmd == KeyMap.recover_files:
                    Init.database.recover_file(path, self.user)
            except Exception as e:
                Logger(e, Logger.inform).log()
                Logger(msg.Info.back_to_routine, Logger.info).log()

    @staticmethod
    def clean_data(data: dict):
        """
        Present the data of the user
        :param data: a dictionary representing the user's data
        """
        data = data.copy()

        del data['file']
        del data['user_id']
        data['file_state'] = ['locked' if state == Database.file_state_locked else 'open' for state in
                              data['file_state']]
        return {data['file_path'][i]: (data['suffix'][i], data['file_state'][i], ) for i in range(len(data['file_path']))}

    def create_show_status_frame(self, *args):
        if self.table_frame is not None:
            return self.table_frame

        frame = tk.Frame(self.button_frame)

        # Add your show status UI elements here
        show_status_label = tk.Label(frame, text="Show Status")
        show_status_label.pack()

        data = Init.database.fetch_user_data(self.user.uid)
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
        Logger(msg.Info.logging_off, Logger.message).log()
        Init.database.lock_all_files(self.user.uid)
        self.parent.switch_frame(LoginWindow(self.parent))

    def delete_user(self):
        Init.database.delete_user(self.user.uid)
        Logger(msg.Info.logging_off, Logger.message).log()

    def destroy(self):
        if self.table_frame is not None:
            self.table_frame.destroy()
        super().destroy()

    def get_window_size(self):
        return self.winfo_width(), self.winfo_height()
