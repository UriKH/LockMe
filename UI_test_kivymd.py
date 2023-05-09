import cv2 as cv
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image
from kivymd.uix.button import MDRaisedButton
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.button import MDRectangleFlatButton


class Login:
    def __init__(self, layout):
        self.logged_in = False
        self.db_image = None

        self.capture = cv.VideoCapture(0)
        self.frame = None
        self.image = Image()

        self.save_img_button = MDRaisedButton(
            text='Take a selfi for later sign in',
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None)
        )
        self.save_img_button.bind(on_press=self.take_screenshot)
        self.layout = layout
        layout.add_widget(self.image)
        layout.add_widget(self.save_img_button)

        Clock.schedule_interval(self.update, 1. / 30)

    def take_screenshot(self, *args):
        self.db_image = self.frame
        self.remove_widget(self.save_img_button)
        self.layout.add_widget(MDRaisedButton(
            text='retake?',
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None)
        ))

        # here pass image into database formatter

    def update(self, *args):
        ret, frame = self.capture.read()
        self.frame = frame

        frame = cv.flip(frame, 1)
        buffer = cv.flip(frame, 0).tobytes()

        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture


class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"
        layout = MDBoxLayout(orientation='vertical')

        self.login = Login(layout)
        return layout


if __name__ == '__main__':
    MainApp().run()
