
from avatar.lib.FileSQLite import FileSQLite
from avatar.lib.AeroSpike import AeroSpike
from avatar.lib.model.convert.product.Customer import Customer

from datetime import date, timedelta


class Avatar(Customer):

    def __init__(self, merchant_id):
        FileSQLite.__init__(self, merchant_id)
        AeroSpike.__init__(self)
        self.pk_date_from = 'date_from_' + str(self.merchant_id)

    # main logic of avatar:
    # convert data from SQLite to AeroSpoke
    def run(self):
        # TODO: only for tests
        # unlock = {'date_from': date.today().strftime("%Y%m%d"), 'in_progress': 0}
        # self.as_row_write(unlock, self.pk_date_from)
        if not self.check_merchant() or not self.get_merchant_files_dir():
            return False
        file_paths = self.get_db_files()
        if len(file_paths) == 0:
            return False
        date_from = False
        process_info = self.as_row_read(self.pk_date_from)
        if bool(process_info) and process_info['in_progress'] == 1:
            self.log('Avatar: merchant\'s already been in work id : ' + str(self.merchant_id))
            return False
        elif bool(process_info) and len(process_info):
            # TODO: just for tests:
            # yesterday = date.today() - timedelta(2)
            # date_from = yesterday.strftime("%Y%m%d")
            date_from = process_info['date_from']
            file_paths = self.remove_finished_files_by_date(file_paths, date_from)
        if len(file_paths) > 1:
            # used, when many files collected for last period, may be was connection error/server unavailable
            united_db_file = self.merge_to_united_db(file_paths)
            if bool(united_db_file):
                # replace by new united DB
                file_paths = []
                file_paths.append(united_db_file)
        if len(file_paths) < 1:
            self.log('Avatar: no new product files for merchant id : ' + str(self.merchant_id))
            return False
        # else:
        # TODO: LOCK current merchant
        lock = {'in_progress': 1}
        self.as_row_write(lock, self.pk_date_from)
        print(self.as_row_read(self.pk_date_from))

        # convert SQLite data to AeroSpike
        self.add_customers_to_db(file_paths)

        # save current date to DB - as last date which will take files ONLY from this date
        # TODO: and UNLOCK current process - anti-double cron launch
        unlock = {'date_from': date.today().strftime("%Y%m%d"), 'in_progress': 0}
        self.as_row_write(unlock, self.pk_date_from)
        print(self.as_row_read(self.pk_date_from))

        return True

    def check_merchant(self):
        if self.merchant_id <= 0:
            self.log('Avatar: merchant_id is NOT defined')
            return False
        return True



