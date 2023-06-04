from UI import *
from initialize import Init
from logger import Logger


def main():
    try:
        Init()
        app = BaseWindow("LockMe")
        app.switch_frame(LoginWindow(app))
        app.mainloop()
    except Exception as e:
        Logger(e, level=Logger.inform).log(main)
    finally:
        del Init.database


if __name__ == "__main__":
    main()
