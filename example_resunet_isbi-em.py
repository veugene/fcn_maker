from __future__ import (print_function,
                        division)
from builtins import input
from collections import OrderedDict
import sys

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader

from torchsample import TensorDataset
from torchsample.modules import ModuleTrainer
from torchsample.regularizers import L2Regularizer
from torchsample.transforms import (RandomAffine,
                                    RandomFlip,
                                    Compose)
from torchsample.callbacks import (EarlyStopping, 
                                   LRScheduler)
from torchsample.metrics import BinaryAccuracy
                                   
from fcn_maker.model import assemble_resunet
from fcn_maker.loss import dice_loss
from fcn_maker.blocks import (tiny_block,
                              basic_block,
                              bottleneck)

'''
Settings.
'''
model_kwargs = OrderedDict((
    ('in_channels', 1),                     # Filled in __main__
    ('num_classes', 1),
    ('num_init_blocks', 2),
    ('num_main_blocks', 3),
    ('main_block_depth', [3, 8, 10, 3, 10, 8, 3]),
    ('init_num_filters', 32),
    ('short_skip', True),
    ('long_skip', True),
    ('long_skip_merge_mode', 'sum'),
    ('main_block', bottleneck),
    ('init_block', tiny_block),
    ('upsample_mode', 'repeat'),
    ('dropout', 0.1),
    # ('normalization', None),                 # Default is batch_normalization
    # ('norm_kwargs', None),                   # Already set by default
    ('init', 'kaiming_normal'),
    ('nonlinearity', 'ReLU'),
    ('ndim', 2),
    ('verbose', True)                          # Set False to silence stdout
    ))
batch_size = 4
cuda_device = 0


'''
Set paths.
'''
data_path = "/tmp/datasets/isbi_2012_em/"
ds_path = {'train-volume': data_path + "train-volume.tif",
           'train-labels': data_path + "train-labels.tif",
           'test-volume': data_path + "test-volume.tif"}


#'''
#Learning rate scheduler.
#'''
#def scheduler(epoch, logs=None):
    #if epoch%200==0 and epoch>0:
        #tmp_lr = (K.get_value(model.optimizer.lr)/10).astype('float32')
        #K.set_value(model.optimizer.lr, tmp_lr)
    #lr = K.get_value(model.optimizer.lr)
    #return np.float(lr)


if __name__=='__main__':
    '''
    Prepare data -- load, standardize, add channel dim., shuffle, split.
    '''
    # Load
    X = tifffile.imread(ds_path['train-volume']).astype(np.float32)
    Y = tifffile.imread(ds_path['train-labels']).astype(np.int64)
    Y[Y==255] = 1
    # Standardize each sample individually (mean center, variance normalize)
    mean = X.reshape(X.shape[0],
                     np.prod(X.shape[1:])).mean(axis=-1)[:,None,None]
    std  = X.reshape(X.shape[0],
                     np.prod(X.shape[1:])).std(axis=-1)[:,None,None]
    X = (X-mean)/std
    # Add channel dim
    X = np.expand_dims(X, axis=1)
    Y = np.expand_dims(Y, axis=1)
    # Shuffle
    R = np.random.permutation(len(X))
    X = X[R]
    Y = Y[R]
    Y = 1-Y
    # Split (26 training, 4 validation)
    X_train = X[:26]
    Y_train = Y[:26]
    X_valid = X[26:]
    Y_valid = Y[26:]
    # Prepare data augmentation and data loaders.
    transform = Compose([RandomAffine(rotation_range=25, shear_range=0.41),
                         RandomFlip(h=True, v=True)])
    ds_train = TensorDataset(X_train, Y_train,
                             co_transform=transform)
    ds_valid = TensorDataset(X_valid, Y_valid)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(ds_valid, batch_size=batch_size)
    
    '''
    Prepare model.
    '''
    model = assemble_resunet(**model_kwargs)
    model.cuda(cuda_device)
    optimizer = torch.optim.RMSprop(params=model.parameters(),
                                    lr=0.001, alpha=0.9)
    trainer = ModuleTrainer(model)
    l2_reg = L2Regularizer(scale=1e-4, module_filter='*conv*')
    trainer.set_regularizers(l2_reg)
    trainer.compile(loss=dice_loss(),
                    optimizer=optimizer,
                    metrics=[BinaryAccuracy()])
    
    
    '''
    Prepare callbacks.
    '''
    callbacks=[]
    #callbacks.append( LRScheduler(scheduler) )
    callbacks.append( EarlyStopping(monitor='val_loss', 
                                    patience=200,
                                    min_delta=0) )
    
    '''
    Train the model.
    '''
    trainer.fit_loader(loader=loader_train,
                       val_loader=loader_valid,
                       num_epoch=500,
                       verbose=1,
                       cuda_device=cuda_device)    
