from avatar.lib.Avatar import Avatar
import time
import datetime

# here will be merchant cycle, which will
# read all files from folder ./a_merchant/merchant_id/product/*
# SQLite3 lib
# load them to aerospike DB
start_time = time.time()

merchant_id = 12345
avatarObject = Avatar(merchant_id)
avatarObject.run()

print('----------------------')
print(
    datetime.datetime.fromtimestamp(
        time.time() - start_time
    ).strftime('%Y-%m-%d %H:%M:%S')
)
