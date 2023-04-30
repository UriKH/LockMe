import sqlite3
import os


class Data:
    path = r'.\databases\database.db'

    def __init__(self, exists=True):
        if not os.path.exists(Data.path):
            exists = False

        self.connection = sqlite3.connect(Data.path)
        self.cursor = self.connection.cursor()

        if not exists:
            self.__init_tables()

        self.data = self.__fetch()

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
            "owner INTEGER,"
            "enc_key TEXT,"
            "file_state INTEGER)"
        )
        self.connection.commit()

    def __fetch(self):
        self.connection.close()
        self.cursor.close()
