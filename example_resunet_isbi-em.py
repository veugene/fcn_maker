from __future__ import (print_function,
                        division)
from builtins import input
from collections import OrderedDict
import sys

import numpy as np
import tifffile
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (EarlyStopping, 
                             LearningRateScheduler)
from keras.optimizers import RMSprop

from lib.model import assemble_resunet
from lib.loss import dice_loss
from lib.blocks import (basic_block_mp,
                        basic_block,
                        bottleneck,
                        get_channel_axis)

'''
Settings.
'''
model_kwargs = OrderedDict((
    ('input_shape', None),                      # Filled in __main__
    ('num_classes', 1),
    ('num_init_blocks', 2),
    ('num_main_blocks', 3),
    ('main_block_depth', [3, 8, 10, 3, 10, 8, 3]),
    ('init_num_filters', 16),
    ('short_skip', True),
    ('long_skip', True),
    ('long_skip_merge_mode', 'sum'),
    ('main_block', bottleneck),
    ('init_block', basic_block_mp),
    ('upsample_mode', 'repeat'),
    ('dropout', 0.1),
    # ('normalization': None),                  # Default is BatchNormalization
    # ('norm_kwargs': None),                    # Already set by default
    ('weight_decay', 0.0001), 
    ('init', 'he_normal'),
    ('nonlinearity', 'relu'),
    ('ndim', 2),
    ('verbose', True)                           # Set False to silence stdout
    ))
batch_size = 1


'''
Set paths.
'''
data_path = "/tmp/datasets/isbi_2012_em/"
ds_path = {'train-volume': data_path + "train-volume.tif",
           'train-labels': data_path + "train-labels.tif",
           'test-volume': data_path + "test-volume.tif"}


'''
Learning rate scheduler.
'''
def scheduler(epoch):
    if epoch%200==0 and epoch>0:
        tmp_lr = (model.optimizer.lr.get_value()/10).astype('float32')
        model.optimizer.lr.set_value(tmp_lr)        
    lr = model.optimizer.lr.get_value()
    return np.float(lr)

'''
Accuracy metric.
'''
def accuracy(y_true, y_pred):
    if model_kwargs['num_classes']==1:
        return K.mean(K.equal(y_true, K.round(y_pred)))
    else:
        return K.mean(K.equal(K.squeeze(y_true, 1),
                              K.argmax(y_pred, axis=-1)))


if __name__=='__main__':
    '''
    Default recursion limit is too low for resnets.
    '''
    sys.setrecursionlimit(99999)

    '''
    Handle input shape for channels_first vs channels_last.
    (eg. theano vs tensorflow)
    '''
    ndim = 2
    channel_axis = get_channel_axis(ndim)
    if channel_axis==1:
        input_shape = (1, 512, 512)
    else:
        input_shape = (512, 512, 1)
    model_kwargs['input_shape'] = input_shape
        
    '''
    Prepare data -- load, standardize, add channel dim., shuffle, split.
    '''
    # Load
    X = tifffile.imread(ds_path['train-volume']).astype(np.float32)
    Y = tifffile.imread(ds_path['train-labels']).astype(np.int32)
    Y[Y==255] = 1
    # Standardize each sample individually (mean center, variance normalize)
    mean = X.reshape(X.shape[0],
                     np.prod(X.shape[1:])).mean(axis=-1)[:,None,None]
    std  = X.reshape(X.shape[0],
                     np.prod(X.shape[1:])).std(axis=-1)[:,None,None]
    X = (X-mean)/std
    # Add channel dim
    X = np.expand_dims(X, axis=channel_axis)
    Y = np.expand_dims(Y, axis=channel_axis)
    # Shuffle
    R = np.random.permutation(len(X))
    X = X[R]
    Y = Y[R]
    # Split (26 training, 4 validation)
    X_train = X[:26]
    Y_train = Y[:26]
    X_val = X[26:]
    Y_val = Y[26:]
    # Prepare data augmentation
    datagen = ImageDataGenerator(rotation_range=25,
                                 shear_range=0.41,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='reflect')
    
    '''
    Prepare model.
    '''
    model = assemble_resunet(**model_kwargs)
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss=dice_loss(target_class=0), 
                  optimizer=optimizer, 
                  metrics=[accuracy])

    '''
    Prepare callbacks.
    '''
    callbacks=[]
    callbacks.append( LearningRateScheduler(scheduler) )
    callbacks.append( EarlyStopping(monitor='val_loss', 
                                    patience=200,
                                    verbose=0,
                                    mode='auto') )
    
    '''
    Train the model.
    '''
    model.fit_generator(datagen.flow(X_train, 
                                     Y_train, 
                                     batch_size=batch_size, 
                                     shuffle=True),
                        steps_per_epoch=len(X_train)//batch_size, 
                        epochs=500,
                        callbacks=callbacks,
                        validation_data=(X_val, Y_val),
                        verbose=2)
    
