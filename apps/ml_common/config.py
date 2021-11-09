from configparser import ConfigParser
import os

class Config:
    DB_TYPE = None
    DB_HOST = None
    DB_PORT = None
    DB_USER = None
    DB_PASSWORD = None
    DB_DATABASE = None

    @classmethod
    def load(self):
        config = ConfigParser()
        dir = os.getcwd()
        with open('config/common.conf') as f:
            config.read_string("[top]\n" + f.read())

        for key in self.__dict__.keys():
            try:
                value = config['top'][str(key).lower()]
                setattr(Config, key, value)
            except Exception:
                pass

Config.load()
