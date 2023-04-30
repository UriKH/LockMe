from cryptography.fernet import Fernet
from logger import Logger
from messages import Messages as msg
import winreg


class Encryption:
    def __init__(self):
        self.key = Encryption.get_key()

    def encrypt_file(self, key):
        pass

    @staticmethod
    def get_key():
        enc_key = None
        try:
            # try getting the key
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Software\\LockMe', 0, winreg.KEY_READ)
            enc_key, value_type = winreg.QueryValueEx(reg_key, 'db_key')
            winreg.CloseKey(reg_key)
        except WindowsError:
            # create the key
            reg_key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, 'Software\\LockMe')

            enc_key = str(Fernet.generate_key())
            value_type = winreg.REG_SZ

            # Set the values for the key
            winreg.SetValueEx(reg_key, 'db_key', 0, value_type, enc_key)
            winreg.CloseKey(reg_key)
        except not WindowsError:
            Logger(msg.Errors.DB_NoEncKey, level=Logger.exception).log()

        if enc_key is None:
            Logger(msg.Errors.BUG, level=Logger.error).log()
        return enc_key
