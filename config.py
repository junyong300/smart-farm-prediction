from configparser import ConfigParser
import os

class Config:
    DB_TYPE = None
    DB_HOST = None
    DB_PORT = None
    DB_USER = None
    DB_PASSWORD = None
    DB_DATABASE = None

    DB_TYPE_LEGACY = None
    DB_HOST_LEGACY = None
    DB_PORT_LEGACY = None
    DB_USER_LEGACY = None
    DB_PASSWORD_LEGACY: str = None
    DB_DATABASE_LEGACY = None

    LOG_LEVEL = 'INFO'

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
        
        # for development
        self.DB_HOST_LEGACY = 'api.farmcloud.kr'
        self.DB_PORT_LEGACY = 3306
        self.DB_USER_LEGACY = 'root'
        self.DB_PASSWORD_LEGACY = ".fc12#$"
        self.DB_DATABASE_LEGACY = 'gr_farm_db'

Config.load()
