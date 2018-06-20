class Debug:
    debug = True

    # def __init__(self):
    #     self.log('')
    #     self.log('Log switched on: ' + str(self.debug))

    def log(self, string):
        if self.debug:
            print(string)
