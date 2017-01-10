from __future__ import print_function

from keras.callbacks import Callback
import numpy as np
import sys
import re
import os
import h5py
import pickle


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
        self.nb_epoch = self.params['nb_epoch']
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
        self.seen = 0


    def on_batch_begin(self, batch, logs={}):
        self.batch = batch
        if self.seen < self.params['nb_sample']:
            self.log_values = {}

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values[k] = logs[k]

        if self.seen < self.params['nb_sample']:
            self.write_log(self.log_values)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch
        for k in self.params['metrics']:
            if k in logs:
                self.log_values[k] = logs[k]
        self.write_log(self.log_values)

class GradientLogger(Callback):
    '''Callback that prints gradient to file.
    '''
    def __init__(self, log_file_name, log_dir, log_write_mode='at'):
        self.log_file_name = log_file_name
        self.log_dir = log_dir
        self.log_write_mode = log_write_mode

    def __del__(self):
        if hasattr(self, 'log_file') and self.log_file is not None:
            self.log_file.close()

    def write_log(self, log_values, description):
        msg = "Epoch {} - {} :: \n".format(self.epoch, description)
        for key, value in enumerate(log_values):
            msg += "\t" + str(value[0])+ ": "+str(value[1]) + "\n"
        print(msg, file=self.log_file)
        self.log_file.flush()

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ self.atoi(c) for c in re.split('(\d+)', text) ]

    def on_train_begin(self, logs={}):
        self.nb_epoch = self.params['nb_epoch']
        self.epoch = 0
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir)
            except:
                print("Failed to create directory for GradientLogger: "
                    "{}".format(self.log_dir))
                raise
        try:
            log_file_path = os.path.join(self.log_dir, self.log_file_name)
            self.log_file = open(log_file_path, self.log_write_mode)
        except:
            print("Failed to open file in GradientLogger: "
                  "{}".format(log_file_path))
            raise

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        self.seen = 0
        self.model.save_weights(self.log_dir + '/tmp1.hdf5', overwrite=True)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch
        self.model.save_weights(self.log_dir + '/tmp2.hdf5', overwrite=True)
        regex=re.compile(".*(convolution).*")

        with h5py.File(self.log_dir + '/tmp1.hdf5') as hf, \
             h5py.File(self.log_dir + '/tmp2.hdf5') as hf2:
            log_epoch = []
            list_of_arrays = [key for key in hf.keys()]
            list_of_convolutions = [m.group(0) for l in list_of_arrays \
                                    for m in [regex.search(l)] if m]
            for x in sorted(list_of_convolutions,
                            key=lambda x: self.natural_keys(x)):
                data_tmp1_b = np.array(hf.get(x)[x + '_b'])
                data_tmp2_b = np.array(hf2.get(x)[x + '_b'])
                data_tmp1_W = np.array(hf.get(x)[x + '_W'])
                data_tmp2_W = np.array(hf2.get(x)[x + '_W'])

                shape = np.shape(data_tmp1_W)
                data_tmp1_W = data_tmp1_W.reshape(shape[0], np.prod(shape[1:]))
                data_tmp2_W = data_tmp2_W.reshape(shape[0], np.prod(shape[1:]))

                W_update = np.abs(data_tmp1_W - data_tmp2_W)
                b_update = np.abs(data_tmp1_b - data_tmp2_b)
                log_gradient = []
                mean_W = W_update.mean(axis=1)
                log_gradient.append(('mean_W', mean_W))
                std_W = W_update.std(axis=1)
                log_gradient.append(('std_W', std_W))
                max_W = W_update.max(axis=1)
                log_gradient.append(('max_W', max_W))
                min_W = W_update.min(axis=1)
                log_gradient.append(('min_W', min_W))
                log_gradient.append(('b_update', b_update))
                log_epoch.append((x, log_gradient))
                self.write_log(log_gradient, x)

        # Pickle the log
        with open(os.path.join(self.log_dir, "Epoch_"+str(self.epoch)+".pkl"),
                  'wb') as f:
            pickle.dump(log_epoch, f)
