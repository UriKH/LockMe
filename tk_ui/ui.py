import tkinter as tk
import tkinter.ttk as ttk
from abc import ABC, abstractmethod


class Window(ABC):
    @abstractmethod
    def get_window_size(self) -> (int, int):
        pass


class BaseWindow(tk.Tk, Window):
    """
    The general app windows configuration
    """
    def __init__(self, title, dims=(800, 600)):
        super().__init__()
        self.title(title)

        style = ttk.Style()
        self.resizable(False, False)
        style.theme_use("clam")
        style.configure("Red.TButton", background="red")
        style.configure("Yellow.TButton", background="yellow")
        self.current_frame = None

        # Set the initial size of the window
        self.geometry(f"{dims[0]}x{dims[1]}")
        self.update()

        # Set minimum size for the window
        self.minsize(*dims)

    def switch_frame(self, new_frame):
        """
        Switch between 2 states of the application: the login and the main windows.
        :param new_frame: the window to switch to
        """
        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = new_frame

        self.current_frame.pack(fill=tk.BOTH, expand=True)

    def get_window_size(self):
        return self.winfo_width(), self.winfo_height()
