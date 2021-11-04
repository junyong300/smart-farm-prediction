import sys
import logging
import pymysql
import pymysql.cursors
import psycopg2
import psycopg2.extras
from psycopg2 import pool

logger = logging.getLogger(__name__)

class DbConn:
    def __init__(self, dbms):
        self.dbms = dbms

    def connect(self, host, port, db, user, passwd):
        if self.dbms == 'mysql':
            conn = pymysql.connect(host=host, port=port, database=db, user=user, passwd=passwd)
            curs = conn.cursor(pymysql.cursors.DictCursor)
        elif self.dbms == 'postgresql':
            postgresql_pool = psycopg2.pool.SimpleConnectionPool(1, 20,
                                                user = user,
                                                password = passwd,
                                                host = host,
                                                port = port,
                                                database = db)
            conn = postgresql_pool.getconn()
            curs = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return curs

if __name__ == '__main__':
    #dbConn = DbConn('mysql')
    #conn = dbConn.connect("192.168.0.221", 3307, "cntd_farm_db", "root", ".fc12#$")
    dbConn = DbConn('postgresql')
    conn = dbConn.connect("192.168.0.229", 5432, "farmconnect", "postgres", ".fc12#$")
