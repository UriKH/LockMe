from UI import *


def main():
    app = BaseWindow("LockMe")
    app.switch_frame(LoginWindow(app))
    app.mainloop()


if __name__ == "__main__":
    main()
