from avatar.lib.AeroSpike import AeroSpike
from avatar.lib.FileSQLite import FileSQLite
from avatar.lib.Debug import Debug
from avatar.lib.Model.Product.Learning import Learning

import pandas as pd
import math


class Customer(AeroSpike, FileSQLite, Debug, Learning):
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
    fields_label = {
        'bought_product'
    }

    key_customer = 'unix_time'

    column_hashes = False

    action_statistic = []

    def __init__(self):
        self.key_for_features += str(self.merchant_id)
        self.table_merchant_actions += str(self.merchant_id)

    def add_actions_to_db(self, file_paths):
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
                    self.table_merchant_actions
                )
            except Exception as e:
                self.log('Customer: cannot save to cluster : ' + str(e))
        return True

    def convert_column_to_db_limit(self, column, save = True):
        if not self.column_hashes:
            # cache data from DB in 1st time
            self.column_hashes = self.as_row_read(
                self.key_for_features,
                self.table_action_features
            )
            if not self.column_hashes:
                self.column_hashes = {}
        if len(column) > self.AS_COLUMN_LENGTH_MAX:
            if self.column_hashes:
                for column_hash, column_value in self.column_hashes.items():
                    if column_value == column:
                        return column_hash
            # else: column is new, need to add
            column_hash = self.convert_to_hash(column)
            self.column_hashes[column_hash] = column
            if save:
                # save new field relation to DB
                self.save_column_to_features(column, column_hash)
            return column_hash
        else:
            if not self.column_hashes.get(column):
                if save:
                    self.save_column_to_features(column, column)
                self.column_hashes[column] = column
        return column

    def save_column_to_features(self, column, column_hash):
        # except not featured columns
        except_columns = self.fields - self.fields_categorized - self.fields_label
        if not column in except_columns:
            return self.as_row_write(
                {column_hash: column},
                self.key_for_features,
                self.table_action_features
            )

        return False

    # TODO: try to replace main has by Murmurhash3, when next time new data imported
    # it should work faster!!!
    # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html#sklearn.feature_extraction.FeatureHasher
    def convert_to_hash(self, string):
        return str(hash(string) % 10 ** self.AS_COLUMN_LENGTH_MAX)


    # TODO: we HAVE create Secondary AeroSpike Index for product_id manually before usage this method!!!
    # https://www.aerospike.com/docs/tools/aql/index_management.html
    # usage in cli:
    # vagrant ssh
    # aql
    # aql> CREATE INDEX index_product_id ON test.cs_merchant_product_actions_12345 (product_id) NUMERIC
    # aql> SHOW INDEXES test
    def calculate_actions_coefficients(self):
        # select all sorted unique products
        query = self.as_select_by_columns(self.table_merchant_actions, None, None, 'product_id')
        products = self.get_unique_products(query)
        # cycle by products
        for productId in products:
            # select actions related just to this product
            query_feature = self.as_select_by_columns(self.table_merchant_actions, 'product_id', productId)

            # do machine learning
            self.query_feature = query_feature
            self.label_name = next(iter(self.fields_label))
            self.label_name_hash = self.convert_column_to_db_limit(self.label_name, False)
            self.allowed_features = self.get_allowed_features(productId)
            Learning.__init__(self)
            estimation = Learning.run(self)

            # machine learning estimation
            total_count = self.get_count_action_by_product_id(productId)
            label_predicted = 0.0
            accuracy_lib = 0.0
            if self.y_pred.any() and self.y_test.any():
                label_predicted = round(sum(self.y_pred) / sum(self.y_test) * 100, 2)
                accuracy_lib = round(Learning.accuracy_lib(Learning, self.y_test, self.y_pred) * 100, 2)

            print('Product_id: ' + str(productId) + '\t' + '\t'
                  + 'Total data: ' + str(total_count) + '\t' + '\t'
                  + 'Bought: ' + str(sum(self.data_label)) + ' = '
                  + str(round(sum(self.data_label) * 100 / total_count, 2)) + '%' + '\t' + '\t' + '\t'
                  + 'BCR = ' + str(round(estimation * 100, 2)) + '%' + '\t' + '\t' + '\t' + '\t'
                  + 'Accuracy = ' + str(accuracy_lib) + '%'
                  + '\t' + '\t' + '\t'
                  + '% "1" in predicted labels/Y: ' + str(label_predicted)
                  + '%' + '\t' + '\t'
                  )

        return True

    def get_unique_products(self, query,):
        products_count = {}
        for (key, metadata, bins) in query.results():
            if not bins['product_id'] in products_count:
                products_count[bins['product_id']] = 1
            else:
                products_count[bins['product_id']] += 1
        self.action_statistic = sorted(products_count.items(), key=lambda kv: kv[1])
        self.action_statistic.reverse()

        # count_uniques = 0
        products_unique = []
        for productId, count in self.action_statistic:
            products_unique.append(productId)
            # count_uniques += 1
        return products_unique

    # calculate and compare count of features/x and current product actions
    def get_allowed_features(self, productId):
        total_count = self.get_count_action_by_product_id(productId)

        column_hashes = self.as_row_read(
            self.key_for_features,
            self.table_action_features
        )
        fields = column_hashes.keys() - self.fields

        if (total_count >= len(fields) * math.log(len(fields))):
            self.is_enough_data_for_dummy = 1
            return fields
        else: # when data is not enough:
            fields_merged = self.fields_categorized.copy()
            fields_merged.update(self.fields_label)

            fields = []
            for field in fields_merged:
                fields.append(self.convert_column_to_db_limit(field))

        if total_count >= len(fields) * math.log(len(fields)):
            self.is_enough_data_for_dummy = 0
            return fields
        else:
            self.is_enough_data_for_dummy = None
            self.log('Customer: error - not enough actions data for such count of features:')
            self.log('Customer: error - count(actions) < count(features) * ln(count(features))')
            self.log('Customer: error - STOP learning for product id = ' + str(productId))
            return []

    def get_count_action_by_product_id(self, productId):
        if (not self.action_statistic):
            self.log('Customer: not statistic/data for current poduct')
            return False
        for product, count in self.action_statistic:
            if product == productId:
                return count
        return False