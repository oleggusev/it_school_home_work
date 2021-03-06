from avatar.lib.Avatar import Avatar
import time

# here will be merchant cycle, which will
# read all files from folder ./a_merchant/merchant_id/product/*
# SQLite3 lib
# load them to aerospike DB
start_time = time.time()

merchant_id = 12345
avatarObject = Avatar(merchant_id)
avatarObject.save_customer_actions()

print('-------- Products save time: --------')
print(str(round((time.time() - start_time) / 60, 2)) + ' mins ')
