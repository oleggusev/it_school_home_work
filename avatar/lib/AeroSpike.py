import aerospike
import sys
from avatar.lib.Debug import Debug


class AeroSpike(Debug):
    # aerospike allows only columns/bins string length no more than 14
    AS_COLUMN_LENGTH_MAX = 13
    # Configure the client
    config = {'hosts': [('172.28.128.3', 3000)]}
    client = False
    namespace = 'test'
    table_product_customers_prefix = 'cs_product_customers_'  # + str(self.merchant_id)
    key_for_hashes_columns_products = 'hashes_columns_for_customers_products'
    table_hashes_columns_relations = 'cs_hashes_columns_relations'

    def __init__(self):
        # Create a client and connect it to the cluster
        try:
            self.client = aerospike.client(self.config).connect()
        except Exception as e:
            error = 'AeroSpike: failed connect to cluster : ' + str(self.config['hosts'])
            Debug.log(self, error)
            sys.exit(error)

    # Write a record
    # client.put(('namespace', 'table', 'primary key # 3'), {
    #   'name': 'John Doe 3',
    #   'age': 333
    # })
    def as_row_write(self, row, primary_key, table='cs_merchants'):
        # Records are addressable via a tuple of (namespace(DB name), set(Table Name), key(Primary Key))
        param = (self.namespace, table, str(primary_key))
        try:
            return self.client.put(param, row)
        except Exception as e:
            if hasattr(e, 'msg'):
                Debug.log(self, 'AeroSpike: as_row_write - ' + e.msg)
            else:
                Debug.log(self, 'AeroSpike: as_row_write - ' + str(e))
            return False

    def as_row_read(self, primary_key, table='cs_merchants'):
        # Records are addressable via a tuple of (namespace(DB name), set(Table Name), key(Primary Key))
        param = (self.namespace, table, primary_key)
        try:
            (key, metadata, row) = self.client.get(param)
        except Exception as e:
            if hasattr(e, 'msg'):
                Debug.log(self, 'AeroSpike: as_row_read - ' + e.msg)
            else:
                Debug.log(self, 'AeroSpike: as_row_read - ' + str(e))
            return False
        return row

    def __del__(self):
        # Close the connection to the Aerospike cluster
        if bool(self.client):
            self.client.close()