from avatar.lib.Avatar import Avatar
import time

# here will be merchant cycle, which will
# read all files from folder ./a_merchant/merchant_id/product/*
# SQLite3 lib
# load them to aerospike DB
start_time = time.time()

merchant_id = 12345
avatarObject = Avatar(merchant_id)
# TODO: we HAVE create Secondary AeroSpike Index for product_id manually before usage this method!!!
avatarObject.calculate_coefficient()

print('------- Coefficients calculate time: -------')
print(str(round((time.time() - start_time) / 60, 2)) + ' mins ')