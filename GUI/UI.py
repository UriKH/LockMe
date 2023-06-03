import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as filedialog
import cv2
from PIL import Image, ImageTk


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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.captured_image = Image.fromarray(rgb_frame)
        image = ImageTk.PhotoImage(self.captured_image)
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
            # Add your login logic here

            # Simulating successful login
            successful_login = True

            if successful_login:
                self.parent.switch_frame(MainWindow(self.parent, self.get_window_size()))
            else:
                print("Login failed")

    def clear_content(self):
        self.frame.pack_forget()
        self.button_frame.pack_forget()
        self.login_button.pack_forget()
        self.label.pack_forget()

    def destroy(self):
        self.camera.release()
        super().destroy()


class MainWindow(tk.Frame):
    def __init__(self, parent, login_window_size):
        super().__init__(parent)
        self.parent = parent
        self.login_window_size = login_window_size

        self.sidebar_frame = ttk.Frame(self, width=150)
        self.sidebar_frame.pack(fill=tk.Y, side=tk.LEFT)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(padx=10, pady=10)

        self.button_map = {
            "Add Files": self.create_add_files_frame,
            "Remove Files": self.create_remove_files_frame,
            "Delete Files": self.create_delete_files_frame,
            "Show Status": self.create_show_status_frame,
            "Lock/Unlock Files": self.create_lock_unlock_files_frame,
            "Recover File": self.create_recover_file_frame
        }

        self.sidebar_buttons = []
        for button_text in self.button_map.keys():
            button = ttk.Button(self.sidebar_frame, text=button_text, command=lambda text=button_text: self.switch_button_frame(text))
            button.pack(pady=5)
            self.sidebar_buttons.append(button)

        self.logout_button = ttk.Button(self.sidebar_frame, text="Logout", style="Yellow.TButton", command=self.logout)
        self.logout_button.pack(pady=10)

        self.delete_user_button = ttk.Button(self.sidebar_frame, text="Delete User", style="Red.TButton")
        self.delete_user_button.pack()

        self.current_button_frame = None

        self.table_frame = None

    def switch_button_frame(self, button_text):
        if self.current_button_frame is not None:
            self.current_button_frame.destroy()

        if button_text == "Show Status":
            self.create_show_status_frame()
        else:
            create_frame_func = self.button_map.get(button_text)
            if create_frame_func:
                self.current_button_frame = create_frame_func()
                self.current_button_frame.pack()
                # Remove the status table if it exists
                if self.table_frame is not None:
                    self.table_frame.destroy()
                    self.table_frame = None

    def choose_files_ui(self, frame):
        file_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, width=50, height=10)
        file_listbox.pack()

        def choose_files():
            selected_paths = filedialog.askopenfilenames()
            if selected_paths:
                for file_path in selected_paths:
                    file_listbox.insert(tk.END, file_path)
                self.run_command(selected_paths)

        choose_files_button = ttk.Button(frame, text="Choose Files", command=choose_files)
        choose_files_button.pack()

    def run_command(self, paths):
        print(paths)

    def create_add_files_frame(self):
        frame = tk.Frame(self.button_frame)

        add_files_label = tk.Label(frame, text="Add Files")
        add_files_label.pack()

        self.choose_files_ui(frame)
        return frame

    def create_remove_files_frame(self):
        frame = tk.Frame(self.button_frame)

        # Add your remove files UI elements here
        remove_files_label = tk.Label(frame, text="Remove Files")
        remove_files_label.pack()

        self.choose_files_ui(frame)
        return frame

    def create_delete_files_frame(self):
        frame = tk.Frame(self.button_frame)

        # Add your delete files UI elements here
        delete_files_label = tk.Label(frame, text="Delete Files")
        delete_files_label.pack()

        self.choose_files_ui(frame)
        return frame

    def create_show_status_frame(self):
        if self.table_frame is not None:
            return self.table_frame

        frame = tk.Frame(self.button_frame)

        # Add your show status UI elements here
        show_status_label = tk.Label(frame, text="Show Status")
        show_status_label.pack()

        table = ttk.Treeview(frame)
        table["columns"] = ("Status")
        table.heading("#0", text="File")
        table.heading("Status", text="Status")

        # Example data
        file_data = {
            "file1.txt": "Modified",
            "file2.txt": "Unmodified",
            "file3.txt": "Modified",
            "file4.txt": "Unmodified"
        }

        for file, status in file_data.items():
            table.insert("", tk.END, text=file, values=(status))

        table.pack(fill=tk.BOTH, expand=True)

        self.table_frame = frame
        self.table_frame.pack()  # Pack the frame into the parent frame

        return frame

    def create_lock_unlock_files_frame(self):
        frame = tk.Frame(self.button_frame)

        # Add your lock/unlock files UI elements here
        lock_unlock_label = tk.Label(frame, text="Lock/Unlock Files")
        lock_unlock_label.pack()

        self.choose_files_ui(frame)
        return frame

    def create_recover_file_frame(self):
        frame = tk.Frame(self.button_frame)

        # Add your recover file UI elements here
        recover_file_label = tk.Label(frame, text="Recover File")
        recover_file_label.pack()

        self.choose_files_ui(frame)

        return frame

    def logout(self):
        self.parent.switch_frame(LoginWindow(self.parent))

    def destroy(self):
        if self.table_frame is not None:
            self.table_frame.destroy()
        super().destroy()

    def get_window_size(self):
        return self.winfo_width(), self.winfo_height()

