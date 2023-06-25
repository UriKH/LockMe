from tk_ui.ui import BaseWindow
from tk_ui.info_window import InfoWindow
from utils.initialize import Init
from utils.logger import Logger


def main():
    try:
        Init()
        app = BaseWindow("LockMe", dims=(800, 600))
        app.switch_frame(InfoWindow(app))  # switch to the Login window
        app.mainloop()
    except Exception as e:
        Logger(e, level=Logger.inform).log()
    finally:
        del Init.database


if __name__ == "__main__":
    main()
