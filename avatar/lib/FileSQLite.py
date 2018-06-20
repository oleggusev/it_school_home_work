import os
import os.path
import glob
import sqlite3
import re
import time
import errno
import pandas as pd

from datetime import datetime, date
from avatar.lib.Debug import Debug


class FileSQLite(Debug):
    # @see Debug class
    # debug = False
    path_to_project = os.path.abspath(os.path.dirname(__file__) + '/..')
    merchant_id = -1
    DIR = 'a_merchant'
    DS = '/'
    FILES_MASK = 'system_????????.log*'
    connect = False
    SQLITE_FILE_NAME = 'unites.db'
    SQLITE_TABLE_NAME = 'cs_product'
    united_db_file = False

    def __init__(self, merchant_id):
        Debug.__init__(self)
        self.merchant_id = merchant_id

    def get_merchant_files_dir(self, directory='product'):
        merchant_path = os.path.join(self.path_to_project, self.DIR + self.DS + str(self.merchant_id)
                                     + self.DS + directory
                                     )
        if not os.path.isdir(merchant_path):
            Debug.log(self, 'FileSQLite: merchant id: ' + str(self.merchant_id) + ' dir: ' + directory + ' is invalid')
            return False

        return merchant_path

    # read merchant files from dir
    def get_db_files(self, mode='product'):
        path_to_data = self.get_merchant_files_dir(mode)
        path_to_files = path_to_data + self.DS + self.FILES_MASK
        file_paths = glob.glob(path_to_files)
        # Debug.log(self, 'Files: ' + str(file_paths))
        if not file_paths:
            Debug.log(self, 'FileSQLite: no files in dir: ' + str(path_to_files))
            return False

        return file_paths

    def remove_finished_files_by_date(self, file_paths, date_from):
        if not bool(date_from):
            return file_paths
        else:
            date_from = self.convert_string_to_unixtime(date_from)

        date_to = self.convert_string_to_unixtime(date.today().strftime('%Y%m%d'))

        for file_path in file_paths[:]:
            parsed = re.findall('system_([0-9]*)\.log*', file_path)
            if len(parsed):
                file_time = self.convert_string_to_unixtime(parsed[0])
                if date_from > file_time or file_time == date_to:
                    # remove all files which less than date_from and equal today
                    # in other worlds: leave just last not processed files AND not today files
                    file_paths.remove(file_path)

        return file_paths

    def convert_string_to_unixtime(self, date_string):
        return time.mktime(datetime.strptime(date_string, "%Y%m%d").timetuple())

    def merge_to_united_db(self, file_paths):
        if not file_paths:
            return False

        if not self.connect:
            self.connect_to_unite_db()

        cursor = self.connect.cursor()
        # read files
        for file_path in file_paths:
            sql_1 = 'attach "' + file_path + '" as toMerge'
            sql_2 = 'BEGIN'
            sql_3 = 'insert into ' + self.SQLITE_TABLE_NAME + ' select * from toMerge.' + self.SQLITE_TABLE_NAME
            sql_4 = 'COMMIT'
            sql_5 = 'detach toMerge'

            try:
                cursor.execute(sql_1)
                cursor.execute(sql_2)
                cursor.execute(sql_3)
                cursor.execute(sql_4)
                cursor.execute(sql_5)
            except sqlite3.Error as e:
                Debug.log(self, 'SQLite: ' + e.msg)

        return self.united_db_file

    def connect_to_unite_db(self):
        self.united_db_file = self.get_merchant_files_dir() + self.DS + self.SQLITE_FILE_NAME
        if os.path.isfile(self.united_db_file):
            try:
                # remove old file and create new, always
                os.remove(self.united_db_file)
            except OSError as e:  # this would be "except OSError, e:" before Python 2.6
                if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                    raise  # re-raise exception if a different error occurred
        self.connect_to_db(self.united_db_file)
        self.create_table(self.SQLITE_TABLE_NAME)
        return self.connect

    def connect_to_db(self, file_path):
        try:
            self.connect = sqlite3.connect(file_path)
        except sqlite3.Error as e:
            Debug.log(self, 'SQLite: ' + str(e))
            return False
        return self.connect

    def create_table(self, table_name):
        cursor = self.connect.cursor()
        sql = '''
        CREATE TABLE IF NOT EXISTS "''' + table_name + '''"
                        (
                            "unix_time" INT PRIMARY KEY,
                            "customer_id" INTEGER,
                            "cs_av_customer" TEXT,
                            "website_id" INTEGER,
                            "product_id" INTEGER,
                            "category_id" INTEGER,
                            "country" TEXT,
                            "city" TEXT,
                            "device_type" TEXT,
                            "device_brand" TEXT,
                            "device_model" TEXT,
                            "is_bot" INTEGER,
                            "month_current" INTEGER,
                            "bought_product" INTEGER
                        )
        '''
        try:
            cursor.execute(sql)
        except sqlite3.Error as e:
            Debug.log(self, 'SQLite: ' + str(e))

        return True

    def sqlite_select(self, sql, return_pandas=True, associated=True):
        if associated:
            self.connect.row_factory = sqlite3.Row
        try:
            if return_pandas:
                return pd.read_sql_query(sql, self.connect)
            else:
                cursor = self.connect.cursor()
                cursor.execute(sql)
                return cursor.fetchall()
        except sqlite3.Error as e:
            Debug.log(self, 'SQLite: ' + str(e))

    def __del__(self):
        self.connect.close()
