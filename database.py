import sqlite3
import os
import bz2 as bz
from encryption import Encryption
from logger import Logger
from messages import Messages as msg
import struct


class Database:
    org_path = r'.\databases\database.db'
    locked_path = r'.\databases\database.locked'

    def __init__(self, exists=True):
        """
        :param exists:
        """
        self.enc_track = Encryption(Database.locked_path, None, 'db', is_db=True)   # track encryption of the database
        if os.path.exists(Database.locked_path):
            self.enc_track.decrypt_file()

        self.connection = sqlite3.connect(Database.org_path)    # connect to the database
        self.cursor = self.connection.cursor()

        self.__init_tables()    # create the tables

        self.users = self.__fetch_users()

    def __del__(self):
        self.connection.close()
        self.enc_track.encrypt_file()

    def __init_tables(self):
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS users("
            "uid INTEGER PRIMARY KEY,"
            "user_embedding TEXT)"
        )
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS files("
            "file_path TEXT PRIMARY KEY,"
            "suffix TEXT,"
            "uid INTEGER,"
            "file TEXT,"
            "state INTEGER)"
        )
        self.connection.commit()

    @staticmethod
    def check_path(path, level):
        if not os.path.exists(path):
            Logger(msg.Errors.file_exists, level=level).log()
            return False
        return True

    @staticmethod
    def _compress_file(data):
        compressor = bz.BZ2Compressor()
        compressed_data = compressor.compress(data) + compressor.flush()
        return compressed_data

    @staticmethod
    def _decompress_file(data):
        decompressed_data = bz.decompress(data).decode()
        return decompressed_data

    @staticmethod
    def _embedding_to_byte(embedding):
        embedding_b = bytearray(struct.pack('512f', *embedding))
        return embedding_b

    @staticmethod
    def _byte_to_embedding(embedding_b):
        embedding = struct.unpack('512f', embedding_b)
        return embedding

    def __fetch_users(self):
        self.cursor.execute("SELECT * FROM users")
        return self.cursor.fetchall()

    def fetch_all_users_data(self):
        self.cursor.execute("SELECT * FROM users")
        return self.cursor.fetchall()

    def _fetch_all_uids(self):
        self.cursor.execute("SELECT uid FROM users")
        return self.cursor.fetchall()

    def fetch_all_files(self, uid):
        self.cursor.execute("SELECT * FROM files WHERE uid = ?", (uid,))
        return self.cursor.fetchall()

    def add_file(self, path, uid):
        if not Database.check_path(path, Logger.warning):
            return False

        key = Encryption.generate_key()
        fd = open(path, 'rb')
        suffix = path.split('.')[-1]
        data = Database._compress_file(fd)

        self.cursor.execute("INSERT INTO files (file_path, suffix, owner, enc_key, file) VALUES (?,?,?,?,?,?)",
                            (path, suffix, uid, key, data))
        self.connection.commit()
        return True

    def create_new_user(self, embedding):
        embedding_b = Database._embedding_to_byte(embedding)
        uids = self._fetch_all_uids()
        max_id = 0
        for uid, _ in uids:     # because uids is of shape (n, )
            if uid > max_id:
                max_id = uid

        self.cursor.execute("INSERT INTO users (uid, user_embedding) VALUES (?,?)",
                            (max_id + 1, embedding_b))
        self.connection.commit()
        return max_id + 1
