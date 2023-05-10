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
    file_state_locked = 0

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
        """
        Check if a file exists and log an error if not
        :param path: the path to the file
        :param level: logging level to pass to the logger
        :return: True if the file exists else False
        """
        if not os.path.exists(path):
            Logger(msg.Errors.file_exists, level=level).log()
            return False
        return True

    @staticmethod
    def _compress_data(data):
        """
        Compress data using the BZ2 compression algorithm
        :param data: data to compress
        :return: the compressed data
        """
        compressor = bz.BZ2Compressor()
        compressed_data = compressor.compress(data) + compressor.flush()
        return compressed_data

    @staticmethod
    def _decompress_data(data):
        """
        Decompress data using the BZ2 compression algorithm
        :param data: compress BZ2 data
        :return: the decompressed data
        """
        decompressed_data = bz.decompress(data).decode()
        return decompressed_data

    @staticmethod
    def _embedding_to_byte(embedding):
        """
        Transform a 512 float list embedding to bytes type
        :param embedding: the embedding vector
        :return: the bytes type embedding
        """
        embedding_b = bytearray(struct.pack('512f', *embedding))
        return embedding_b

    @staticmethod
    def _byte_to_embedding(embedding_b):
        """
        Transform bytes type embedding to a 512 float list
        :param embedding_b: the bytes type embedding
        :return: the vectorized embedding
        """
        embedding = struct.unpack('512f', embedding_b)
        return embedding

    def fetch_users(self):
        """
        Fetch all user IDs and their correlated face embeddings from the DB
        :return: the retrieved data
        """
        self.cursor.execute("SELECT * FROM users")
        return self.cursor.fetchall()

    def _fetch_all_ids(self):
        """
        Fetch all user IDs from the DB
        :return: the retrieved IDs
        """
        self.cursor.execute("SELECT uid FROM users")
        return self.cursor.fetchall()

    def fetch_user_data(self, uid):
        """
        Retrieve the user's data
        :param uid: the current user ID
        :return: the user's data as a dictionary:
            {'file_path': [...], 'suffix': [...], 'user_id': [...], 'file': [...], 'checksum': [...],
            'file_state': [...]}
        """
        self.cursor.execute("SELECT file_path, suffix, uid, file, checksum, file_state FROM files WHERE uid = ?", (uid,))
        data = self.cursor.fetchall()
        data_dict = {'file_path': [], 'suffix': [], 'user_id': [], 'file': [], 'checksum': [], 'file_state': []}
        for row in data:
            data_dict['file_path'].append(row[0])
            data_dict['suffix'].append(row[1])
            data_dict['user_id'].append(row[2])
            data_dict['file'].append(row[3])
            data_dict['checksum'].append(row[4])
            data_dict['file_state'].append(row[5])
        return data_dict

    def add_file(self, path, user):
        """
        Add a file to the DB
        :param path: the path to the file to be added
        :param user: a User object
        :return: True if file was successfully added to the DB
        """
        # check if the file is already in the database
        self.cursor.execute("SELECT file_path, uid FROM files")
        res = self.cursor.fetchall()
        for row in res:
            if path == row[0]:
                Logger(msg.Errors.failed_insertion + f' - owner ID: {row[1]}', Logger.inform).log()
                return False

        if '.' in path:
            suffix = path.split('.')[-1]
        else:
            suffix = 'no suffix'
        with open(path, 'rb') as fd:
            data = fd.read()

        checksum = Database.checksum(data)
        key = self.get_user_embedding_as_key(user.uid)
        file_enc = Encryption(path, key, suffix)
        enc_data = file_enc.encrypt_file()
        comp_data = Database._compress_data(enc_data)

        self.cursor.execute(
            "INSERT INTO files (file_path, suffix, uid, file, checksum, file_state) VALUES (?, ?, ?, ?, ?, ?)",
            (path, suffix, user.uid, comp_data, checksum, Database.file_state_locked))
        self.connection.commit()
        Logger(msg.Info.file_added + f' {path}', Logger.info).log()
        return True

    def remove_file(self, path, user):
        """
        Remove a file from the DB
        :param path: the path to the file to remove
        :param user: a User object
        :return: True if the file removed successfully
        """
        # check if action is valid on the file
        self.cursor.execute("SELECT file_path, uid, suffix, checksum, file, file_state FROM files")
        res = self.cursor.fetchall()
        suffix = None
        file_state = None
        comp_file = None
        checksum = None

        found = False
        for row in res:
            if path == row[0]:
                if row[1] != user.uid:
                    Logger(msg.Errors.access_denied + f' to file {path}', Logger.inform).log()
                    return False
                found = True
                suffix = row[2]
                checksum = row[3]
                comp_file = row[4]
                file_state = row[5]
                break
        if not found:
            Logger(msg.Errors.failed_removal, Logger.inform).log()
            return False

        # decrypt the file if encrypted
        if file_state == Database.file_state_locked:
            key = self.get_user_embedding_as_key(user.uid)
            file_enc = Encryption(path, key, suffix)
            file_enc.decrypt_file()

        # read contents and calculate checksum
        with open(path, 'rb') as fd:
            raw_data = fd.read()
        if Database.checksum(raw_data) != checksum:
            # recover the original file
            with open(path, 'wb') as fd:
                recovered_data = Encryption.decrypt_data(comp_file, user.embedding)
                path.write(recovered_data)
                Logger(msg.Info.file_recovered + f' {path}', Logger.info).log()

        # delete the file from the database
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

    @staticmethod
    def checksum(data):
        """
        Data to compute checksum on
        :param data: the data in the file as values
        :return: sum of file
        """
        val = 0
        for char in data:
            val += char
        Logger(msg.Info.checksum + f' {val}', Logger.info).log()
        return val

    def get_user_embedding_as_key(self, uid):
        """
        Retrieve the user face embedding from the DB and generate the key from it
        :param uid: the user ID
        :return: the generated fernet key
        """
        self.cursor.execute("SELECT user_embedding FROM users WHERE uid = ?", (uid,))
        embedding_b = self.cursor.fetchall()[0][0]
        embedding = Database._byte_to_embedding(embedding_b)
        key = Encryption.key_from_embedding(embedding)
        return key
