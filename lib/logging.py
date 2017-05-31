from __future__ import print_function
from keras.callbacks import Callback
import numpy as np


class FileLogger(Callback):
    '''Callback that prints loss and metrics to file.
    '''
    def __init__(self, log_file_path, log_write_mode='at'):
        self.log_file_path = log_file_path
        self.log_write_mode = log_write_mode

    def __del__(self):
        if hasattr(self, 'log_file') and self.log_file is not None:
            self.log_file.close()

    def write_log(self, log_values):
        if self.log_file is not None:
            msg = "Epoch {} - batch {} :: ".format(self.epoch, self.batch)
            for i, key in enumerate(sorted(self.log_values.keys())):
                if i>0:
                    msg += " - "
                msg += str(key)+": "+str(self.log_values[key])
            print(msg, file=self.log_file)
            self.log_file.flush()

    def on_train_begin(self, logs={}):
        self.epochs = self.params['epochs']
        self.epoch = 0
        if self.log_file_path is not None:
            try:
                self.log_file = open(self.log_file_path, self.log_write_mode)
            except:
                print("Failed to open file in FileLogger: "
                      "{}".format(self.log_file_path))
                raise

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs={}):
        self.batch = batch
        self.log_values = {}

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        for k in self.params['metrics']:
            if k in logs:
                self.log_values[k] = logs[k]
        self.write_log(self.log_values)

    def on_epoch_end(self, epoch, logs={}):
        pass
