import sqlite3
import os
import bz2 as bz
from encryption import Encryption
from logger import Logger
from messages import Messages as msg


class Database:
    org_path = r'.\databases\database.db'
    locked_path = r'.\databases\database.locked'

    def __init__(self, exists=True):
        if not (Database.__check_path(Database.org_path, Logger.info) or
                Database.__check_path(Database.locked_path, Logger.info)):
            exists = False

        self.enc_track = Encryption(Database.locked_path, None, 'db')
        if os.path.exists(Database.locked_path):
            self.enc_track.decrypt_file()

        self.connection = sqlite3.connect(Database.org_path)
        self.cursor = self.connection.cursor()

        if not exists:
            self.__init_tables()

        self.users = self.__fetch_users()

    def __init_tables(self):
        self.cursor.execute(
            "CREATE TABLE users("
            "uid INTEGER,"
            "user_encoding TEXT,"
            "username TEXT,"
            "pwd TEXT,"
            "PRIMARY KEY (uid, username))"
        )
        self.cursor.execute(
            "CREATE TABLE files("
            "file_path TEXT PRIMARY KEY,"
            "suffix TEXT,"
            "owner INTEGER,"
            "enc_key TEXT,"
            "file TEXT)"
        )
        self.connection.commit()

    def __fetch_users(self):
        self.cursor.execute("SELECT * FROM users")
        return self.cursor.fetchall()

    @staticmethod
    def __check_path(path, level):
        if not os.path.exists(path):
            Logger(msg.Errors.file_exists, level=level).log()
            return False
        return True

    @staticmethod
    def __compress_file(data):
        compressor = bz.BZ2Compressor()
        compressed_data = compressor.compress(data) + compressor.flush()
        return compressed_data

    @staticmethod
    def __decompress_file(data):
        decompressed_data = bz.decompress(data).decode()
        return decompressed_data

    def fetch_all(self, uid):
        self.cursor.execute("SELECT * FROM files WHERE uid = ?", (uid,))
        return self.cursor.fetchall()

    def add_file(self, path, uid):
        if not Database.__check_path(path, Logger.warning):
            return False

        key = Encryption.generate_key()
        fd = open(path, 'rb')
        suffix = path.split('.')[-1]
        data = Database.__compress_file(fd)

        self.cursor.execute("INSERT INTO files (file_path, suffix, owner, enc_key, file) VALUES (?,?,?,?,?,?)",
                            (path, suffix, uid, key, data))
        self.connection.commit()
        return True
