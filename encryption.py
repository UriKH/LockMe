import os
import numpy as np
from cryptography.fernet import Fernet
import winreg
import base64

from utils.logger import Logger
from utils.messages import Messages as msg
from model.SNN import Net


class Encryption:
    """
    Handling encryption operations on files
    """

    def __init__(self, path, key, suffix, is_db=False):
        self.key = key if not is_db else Encryption.__retrieve_registry_key(path='Software\\LockMe', key_name='db_key')
        self.fernet = Fernet(self.key)

        clean_path = '.'.join(path.split('.')[:-1])
        self.locked_path = '.'.join([clean_path, 'locked'])
        self.org_path = '.'.join([clean_path, suffix])

    def encrypt_file(self, log: bool = True):
        """
        Encrypt the file
        :return: the encrypted file contents
        """
        with open(self.org_path, 'rb') as file:
            raw_data = file.read()
            enc_data = self.fernet.encrypt(raw_data)

        with open(self.org_path, 'wb') as f:
            f.write(enc_data)
            if log:
                Logger(f'{msg.Info.file_encrypt} - path is {self.locked_path}', Logger.warning).log()

        os.rename(self.org_path, self.locked_path)
        return enc_data

    def decrypt_file(self, log: bool = True):
        """
        Decrypt the file
        :return: the decrypted contents
        """
        with open(self.locked_path, 'rb') as file:
            enc_data = file.read()
            raw_data = self.fernet.decrypt(enc_data)

        with open(self.locked_path, 'wb') as f:
            f.write(raw_data)
            if log:
                Logger(f'{msg.Info.file_decrypt} - path is {self.org_path}', Logger.warning).log()

        os.rename(self.locked_path, self.org_path)
        return raw_data

    @staticmethod
    def _generate_key():
        """
        Generate a fernet encryption key
        :return: the key
        """
        return Fernet.generate_key()

    @staticmethod
    def __retrieve_registry_key(path: str, key_name: str):
        """
        Retrieve or generate the database encryption key and save it in the registry
        :return: The database encryption key
        """
        enc_key = None
        try:
            # try getting the key
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_READ)
            enc_key, value_type = winreg.QueryValueEx(reg_key, key_name)
            winreg.CloseKey(reg_key)
        except WindowsError:
            Logger(msg.Info.new_db_key, Logger.info).log()

            # create the key
            reg_key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, path)

            enc_key = Encryption._generate_key().decode('utf-8')
            value_type = winreg.REG_SZ

            # Set the values for the key
            winreg.SetValueEx(reg_key, key_name, 0, value_type, enc_key)
            winreg.CloseKey(reg_key)
        except not WindowsError:
            Logger(msg.Errors.db_no_enc_key, level=Logger.exception).log()

        if enc_key is None:
            Logger(msg.Errors.BUG, level=Logger.error).log()
        return enc_key.encode('utf-8')

    @staticmethod
    def key_from_embedding(embedding: list):
        """
        Generate a 128 bits key from float vector
        :param embedding: A float vector representing an image
        :return: The key generated from the vector
        """
        if len(embedding) < Net.embedding_size:
            Logger(msg.Errors.BUG, Logger.exception).log()

        step = int(Net.embedding_size / 128)
        for i in range(128):
            embedding[i] = np.sum(np.array(embedding[i * step: i * step + step]))
        embedding = embedding[: 128]

        # embedding = [embedding[i] + embedding[i+1] for i in range(0, 4096, 32)]
        bin_embedding = (np.array(embedding) > 0.5).astype(int)
        binary_string = ''.join(str(bit) for bit in bin_embedding)
        integer = int(binary_string, 2)
        bytes_integer = integer.to_bytes(32, byteorder='big')
        key = base64.urlsafe_b64encode(bytes_integer)
        return key

    @staticmethod
    def encrypt_data(data: bytes, key):
        """
        Encrypt some data with a fernet key
        :param data: the data
        :param key: the fernet key
        :return: the encrypted data
        """
        fernet = Fernet(key)
        return fernet.encrypt(data)

    @staticmethod
    def decrypt_data(data: bytes, key):
        """
        Decrypt some data with a fernet key
        :param data: the data
        :param key: the fernet key
        :return: the decrypted data
        """
        fernet = Fernet(key)
        return fernet.decrypt(data)
