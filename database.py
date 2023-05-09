import sqlite3
import os
import bz2 as bz
import struct

from encryption import Encryption
from logger import Logger
from messages import Messages as msg


class Database:
    # TODO: implement all database commands
    org_path = r'.\databases\database.db'
    locked_path = r'.\databases\database.locked'
    file_state_open = 1
    file_state_close = 0

    def __init__(self):
        self.enc_track = Encryption(Database.locked_path, None, 'db', is_db=True)   # track encryption of the database
        if os.path.exists(Database.locked_path):
            self.enc_track.decrypt_file()

        self.connection = sqlite3.connect(Database.org_path)    # connect to the database
        self.cursor = self.connection.cursor()

        self.__init_tables()    # create the tables

        self.users = self.fetch_users()

    def __del__(self):
        """
        Encrypt the DB before termination
        """
        self.connection.close()
        self.enc_track.encrypt_file()

    def __init_tables(self):
        """
        Initiate the database's tables if needed
        """
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
            "checksum INTEGER,"
            "file_state INTEGER)"
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

    @staticmethod
    def checksum(data):
        val = 0
        for char in data:
            val += char
        Logger(msg.Info.checksum + f' {val}', Logger.info).log()
        return val

    def fetch_users(self):
        self.cursor.execute("SELECT * FROM users")
        return self.cursor.fetchall()

    def _fetch_all_ids(self):
        self.cursor.execute("SELECT uid FROM users")
        return self.cursor.fetchall()

    def fetch_user_data(self, uid):
        """
        Retrieve the user's data
        :param uid: the current user ID
        :return: the user's data
        """
        self.cursor.execute("SELECT * FROM files WHERE uid = ?", (uid,))
        return self.cursor.fetchall()

    def add_file(self, path, uid):
        """
        Add a file to the DB
        :param path: the path to the file to be added
        :param uid: the user ID
        :return: True if file was successfully added to the DB
        """
        # check if the file is already in the database
        self.cursor.execute("SELECT file_path, uid FROM files")
        res = self.cursor.fetchall()
        for row in res:
            if path == row[0]:
                Logger(msg.Errors.failed_insertion + f' - owner ID: {row[1]}', Logger.inform).log()
                return False

        with open(path, 'rb') as fd:
            data = fd.read()
            suffix = path.split('.')[-1]
            checksum = Database.checksum(data)
            data = Database._compress_file(data)

        self.cursor.execute(
            "INSERT INTO files (file_path, suffix, uid, file, checksum, file_state) VALUES (?, ?, ?, ?, ?, ?)",
            (path, suffix, uid, data, checksum, Database.file_state_open))
        self.connection.commit()
        Logger(msg.Info.file_added + f' {path}', Logger.info).log()
        return True

    def remove_file(self, path, uid):
        """
        Remove a file from the DB
        :param path: the path to the file to remove
        :param uid: the current user ID
        :return: True if the file removed successfully
        """
        # TODO: decrypt file if it is encrypted
        # check if the file is already in the database
        self.cursor.execute("SELECT file_path, uid FROM files")
        res = self.cursor.fetchall()

        found = False
        for row in res:
            if path == row[0]:
                if row[1] != uid:
                    Logger(msg.Errors.access_denied + f' to file {path}', Logger.inform).log()
                    return False
                found = True
                break
        if not found:
            Logger(msg.Errors.failed_removal, Logger.inform).log()
            return False

        self.cursor.execute("DELETE FROM files WHERE file_path = ?", (path,))
        self.connection.commit()
        Logger(msg.Info.file_removed + f' {path}', Logger.info).log()
        return True

    def create_new_user(self, embedding):
        """
        ADD a new user to the system
        :param embedding: a 512 embedding vector of the user's face
        :return: the new user's ID
        """
        embedding_b = Database._embedding_to_byte(embedding)
        uids = self._fetch_all_ids()
        max_id = 0
        for row in uids:     # because uids is of shape (n, )
            uid = row[0]
            if uid > max_id:
                max_id = uid

        self.cursor.execute("INSERT INTO users (uid, user_embedding) VALUES (?,?)",
                            (max_id + 1, embedding_b))
        self.connection.commit()
        return max_id + 1
