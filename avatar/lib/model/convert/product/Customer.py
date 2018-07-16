from avatar.lib.AeroSpike import AeroSpike
from avatar.lib.FileSQLite import FileSQLite
from avatar.lib.Debug import Debug

import pandas as pd


class Customer(AeroSpike, FileSQLite, Debug):
    fields = {
        'unix_time',
        'customer_id',
        'cs_av_customer',
        'website_id',
        'product_id',
        'category_id',
        'country',
        'city',
        'device_type',
        'device_brand',
        'device_model',
        'is_bot',
        'month_current',
        'bought_product'
    }
    fields_categorized = {
        'category_id',
        'country',
        'city',
        'device_type',
        'device_brand',
        'device_model',
        'month_current'
    }
    key_customer = 'unix_time'

    column_hashes = False

    def add_customers_to_db(self, file_paths):
        if len(file_paths):
            file_paths = file_paths[0]

        if not self.connect_to_db(file_paths):
            return False
        products = self.get_sorted_products()
        if products.empty:
            return False
        dummy_products = pd.get_dummies(products, columns=self.fields_categorized)
        # save both fields dummy and original to DB - for later CV
        for field in self.fields_categorized:
            # add original columns
            dummy_products[field] = products[field]
        # TODO: just for tests
        # dummy_products.to_csv(self.get_merchant_files_dir() + self.DS + 'a111.csv')
        self.save_to_db(dummy_products)
        return True

    def get_sorted_products(self):
        sql = 'SELECT * FROM "' + self.SQLITE_TABLE_NAME + '"'
        sql += ' WHERE is_bot = 0 AND cs_av_customer IS NOT NULL AND product_id IS NOT NULL'
        sql += ' ORDER BY product_id, category_id'
        products = self.sqlite_select(sql)
        return products

    # TRUNCATE set:
    # vagrant ssh
    # asinfo -v "truncate:namespace=test;set=cs_product_customers"
    def save_to_db(self, products):
        record = {}
        products_list = products.to_dict('records')
        for value_dict in products_list:
            try:
                for column in value_dict:
                    record[self.convert_column_to_db_limit(column)] = value_dict[column]
                self.as_row_write(
                    record,
                    str(value_dict[self.key_customer]),
                    self.table_product_customers_prefix + str(self.merchant_id)
                )
            except Exception as e:
                self.log('Customer: cannot save to cluster : ' + str(e))
        return True

    def convert_column_to_db_limit(self, column):
        if len(column) > self.AS_COLUMN_LENGTH_MAX:
            if not self.column_hashes:
                # cache data from DB
                self.column_hashes = self.as_row_read(
                    self.key_for_hashes_columns_products,
                    self.table_hashes_columns_relations
                )
            if self.column_hashes:
                for column_hash, column_value in self.column_hashes.items():
                    if column_value == column:
                        return column_hash
            # else: column is new, need to add
            column_hash = str(hash(column) % 10 ** self.AS_COLUMN_LENGTH_MAX)
            # save new field relation to DB
            self.as_row_write(
                {column_hash: column},
                self.key_for_hashes_columns_products,
                self.table_hashes_columns_relations
            )
            return column_hash
        return column
