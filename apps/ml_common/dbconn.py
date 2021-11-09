import sys
import logging
import asyncio
from databases import Database

'''
import pymysql
import pymysql.cursors
import psycopg2
import psycopg2.extras
from psycopg2 import pool
'''


logger = logging.getLogger(__name__)

class DbConn:
    async def connect(self, type, host, port, db, user, password):
        conn = Database(F'{type}://{host}:{port}/{db}', user=user, password=password)
        await conn.connect()

        '''
        postgresql_pool = psycopg2.pool.SimpleConnectionPool(1, 20,
                                            user = user,
                                            password = passwd,
                                            host = host,
                                            port = port,
                                            database = db)
        conn = postgresql_pool.getconn()
        curs = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            '''
#        return curs

        return conn

if __name__ == '__main__':
    #dbConn = DbConn('mysql')
    #conn = dbConn.connect("192.168.0.221", 3307, "cntd_farm_db", "root", ".fc12#$")
    dbConn = DbConn('postgresql')
    conn = asyncio.run(dbConn.connect("192.168.0.229", 5432, "farmconnect", "postgres", ".fc12#$"))
    
