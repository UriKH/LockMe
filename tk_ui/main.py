from tk_ui.UI import *
from utils.initialize import Init
from utils.logger import Logger


def main():
    try:
        Init()
        app = BaseWindow("LockMe")
        app.switch_frame(LoginWindow(app))  # switch to the Login window
        app.mainloop()
    except Exception as e:
        Logger(e, level=Logger.inform).log(main)
    finally:
        del Init.database


if __name__ == "__main__":
    main()
