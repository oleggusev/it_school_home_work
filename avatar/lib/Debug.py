class Debug:
    debug = True

    write_to_file_log = False
    file = None

    def __init__(self):
        #     self.log('')
        #     self.log('Log switched on: ' + str(self.debug))
        if self.write_to_file_log:
            self.file = open("logic_regression.log", "a+")

    def log(self, string):
        if self.debug:
            print(string)
        if self.write_to_file_log:
            print(string, file=self.file)

    def printDictionary(self, dict):
        for i in dict.items():
            if self.debug:
                print("%s\t%s"%(i[0], i[1]))
            if self.write_to_file_log:
                print("%s\t%s" % (i[0], i[1]), file=self.file)

    def write_to_file(self, string):
        if self.write_to_file_log:
            self.file.write(string)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.write_to_file_log and self.file:
            self.file.close()