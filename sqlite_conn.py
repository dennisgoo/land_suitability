# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:53:11 2018
A class for building sqlite connection
@author: guoj
"""


import sqlite3
from sqlite3 import Error

class Sqlite_connection(object):
    
    def __init__(self, db_file):
        self.db_file = db_file
        
    def __enter__(self):
        """ create a database connection to the SQLite database
            specified by the db_file
        :param db_file: database file
        :return: Connection object or None
        """
        try:
            self.conn = sqlite3.connect(self.db_file)
            self.conn.row_factory = sqlite3.Row
            return self.conn.cursor()
        except Error as e:
            print(e)
        
        return None
    
    def __exit__(self, type, value, traceback):
        self.conn.commit()
        self.conn.close()