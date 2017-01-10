"""
all proper (not wide resnet style) shortcuts;
"""

import sys;
sys.setrecursionlimit(40000)
sys.path.append("../")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Model
from keras.engine import Input
from keras.layers import (Lambda,
                          Dense,
                          Convolution2D,
                          BatchNormalization,
                          MaxPooling2D,
                          AveragePooling2D,
                          Activation,
                          Flatten,
                          merge)
from keras import backend as K
from lib.mem_resnet import stackenblochen, bn_relu_conv, _l2
from lib.logging import FileLogger

import pickle
import os
import shutil
import numpy as np
from copy import copy
from collections import OrderedDict
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (EarlyStopping, 
                             LearningRateScheduler, 
                             ModelCheckpoint)
from keras.optimizers import RMSprop, SGD




# -----------------------------------------------------------------------------
model_kwargs = OrderedDict((
    ('memory', True),
    ('residual', True),
    ('merge_memory', False),
    ('k', 10),    # width factor
    ('r', 4),     # num reptitions per block
    ('dropout', 0.),
    ('weight_decay', 0.0005),
    ('update_bias', 0),
    ('read_bias', 0),
    ('momentum', 0.95)
    ))
P = OrderedDict((
    ('num_classes', 100),
    ('batch_size', 128),
    ('nb_epoch', 400),
    ('lr_schedule', ['constant', {0: 0.1, 60: 0.02, 120: 0.004, 160: 0.0008}]),
    ('patience', 30),
    ('stop_early', False),
    ('optimizer_type', 'SGD'), # 'RMSprop' or 'SGD'
    ('validation_set', None)  # Size of validation set or None to use test
    ))
plot_model = False

experiment_ID = "example-memcnn"
ds_dir = "/tmp/datasets/cifar100/cifar-100-python"
results_dir = "/tmp/results"

np.random.seed(1337)
# -----------------------------------------------------------------------------




def GlobalAveragePooling2D(input):
    return Lambda(function=lambda x: K.mean(x.flatten(3), axis=2),
                  output_shape=lambda s: s[:2])(input)

def assemble_model(input, residual=True, memory=True, k=4, r=4,
                   merge_memory=False, dropout=0., weight_decay=None,
                   update_bias=None, read_bias=None, momentum=0.9):
    
    # First conv
    layer1 = Convolution2D(nb_filter=16, nb_row=3, nb_col=3, subsample=(1, 1),
                           init='he_normal', border_mode='same', bias=False,
                           W_regularizer=_l2(weight_decay))(input)
    layer1 = BatchNormalization(mode=0, axis=1, momentum=momentum)(layer1)
    layer1 = Activation('relu')(layer1)
    
    if memory:
        inputs = (layer1, layer1)
    else:
        inputs = (layer1, None)
        
    # 3 blocks
    shared_kwargs = {'repetitions' : r,
                     'residual' : residual,
                     'memory' : memory,
                     'dropout' : dropout,
                     'update_bias' : update_bias,
                     'read_bias' : read_bias,
                     'weight_decay' : weight_decay,
                     'momentum' : momentum}
    block_1 = stackenblochen(16*k, subsample=False, **shared_kwargs)(*inputs)
    block_2 = stackenblochen(32*k, subsample=True, **shared_kwargs)(*block_1)
    block_3 = stackenblochen(64*k, subsample=True, **shared_kwargs)(*block_2)
    
    # Feature output (combine memory?)
    if memory and merge_memory:
        feature_output = merge([block_3[0], block_3[1]],
                               mode='concat', concat_axis=1)
    else:
        feature_output = block_3[0]
        
    # Final batch norm + nonlinearity
    feature_output = BatchNormalization(mode=0, axis=1, momentum=momentum)(feature_output)
    feature_output = Activation('relu')(feature_output)
        
    # Average pooling
    avgpool = GlobalAveragePooling2D(feature_output)
    #avgpool = Flatten()(AveragePooling2D(pool_size=(8, 8), strides=(1, 1), border_mode='valid')(feature_output))
    
    sm = Dense(P['num_classes'], init='he_normal', activation='softmax',
               W_regularizer=_l2(weight_decay))(avgpool)
    model = Model(input=input, output=sm)
    return model 


def unpickle(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def onehot(labels, num_classes):
    onehot_labels = np.zeros((len(labels), num_classes), dtype=np.int32)
    for i, l in enumerate(labels):
        onehot_labels[i,l] = 1
    return onehot_labels


if __name__ == '__main__':
    # Print settings to screen
    print("Experiment:", experiment_ID)
    print("")
    for key in model_kwargs.keys():
        print(key, ":", model_kwargs[key])
    print("")
    for key in P.keys():
        print(key, ":", P[key])
    
    
    # Set up experiment directory
    append_ID = ""
    if model_kwargs['memory'] and model_kwargs['merge_memory']:
        append_ID += "M"
    elif model_kwargs['memory']:
        append_ID += "m"
    if model_kwargs['residual']:
        append_ID += "r"
    if append_ID != "":
        experiment_ID += "_"+append_ID
        experiment_ID += "-"+str(model_kwargs['k'])+"-"+str(model_kwargs['r'])

    results_dir = os.path.join(results_dir, experiment_ID)
    if os.path.exists(results_dir):
        print("")
        print("WARNING! Results directory exists: \"{}\"".format(results_dir))
        write_into = None
        while write_into not in ['y', 'n', 'r', '']:
            write_into = str.lower(input( \
                            "Write into existing directory? [y/N/r(eplace)]"))
        if write_into in ['n', '']:
            print("Aborted")
            sys.exit()
        if write_into=='r':
            shutil.rmtree(results_dir)
            print("WARNING: Deleting existing results directory.")
        print("")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    fn = sys.argv[0].rsplit('/', 1)[-1]
    shutil.copy(sys.argv[0], os.path.join(results_dir, fn))
    
    
    # Assemble model
    input = Input(shape=(3, 32, 32))
    model = assemble_model(input, **model_kwargs)
    if plot_model:
        from keras.utils.visualize_util import plot
        plot(model, to_file=os.path.join(results_dir, "model.png"))
        
    yaml_string = model.to_yaml()
    open(os.path.join(results_dir, "model_" +
                      str(experiment_ID) +
                      ".yaml"), 'w').write(yaml_string)
    
    print("number of parameters : ", model.count_params())
    
    
    # Load data
    print("")
    print("--------------------------------------------------")
    print("Preparing data")
    
    data_batches = []
    label_batches = []
    ds_train = unpickle(os.path.join(ds_dir, "train"))
    X_train = ds_train['data']
    Y_train = onehot(ds_train['fine_labels'], P['num_classes'])
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='reflect')
    
    if P['validation_set'] is None:
        # Load test set
        ds_test = unpickle(os.path.join(ds_dir, "test"))
        X_test = ds_test['data']
        Y_test = onehot(ds_test['fine_labels'], P['num_classes'])
    else:
        # Split out a validation set
        R = np.random.permutation(len(X_train))
        valid_idx = R[:P['validation_set']]
        train_idx = R[P['validation_set']:]
        X_test = X_train[valid_idx]
        Y_test = Y_train[valid_idx]
        X_train = X_train[train_idx]
        Y_train = Y_train[train_idx]
    
    # Reshape to 2D
    X_train = X_train.reshape(-1,3,32,32)
    X_test = X_test.reshape(-1,3,32,32)
    
    # Pixel-wise mean center
    pixel_mean = np.mean(X_train, axis=0)
    X_train = X_train.astype(np.float32) - pixel_mean
    X_test = X_test.astype(np.float32) - pixel_mean
    
    # Compile model
    if P['optimizer_type'] == 'RMSprop':
        optimizer = RMSprop(lr=P['lr_schedule'][1][0], rho=0.9, epsilon=1e-08)
    elif P['optimizer_type'] == 'SGD':
        optimizer = SGD(lr=P['lr_schedule'][1][0], momentum=0.9, decay=0.0,
                        nesterov=True)
    else:
        optimizer = None
    from theano import tensor as T
    #def cce(target, output):
        ##output /= output.sum(axis=-1, keepdims=True)
        ##_EPSILON = 10e-8
        ##output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
        #return T.nnet.categorical_crossentropy(output, target)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    # RUN training
    print("Training")
    # Set up callbacks
    callbacks = []
    
    if P['lr_schedule'][0]=='constant':
        def scheduler(epoch):
            if epoch in P['lr_schedule'][1]:
                model.optimizer.lr.set_value(P['lr_schedule'][1][epoch])        
            lr = model.optimizer.lr.get_value()
            return np.float(lr) 
    
        change_lr = LearningRateScheduler(scheduler)
        callbacks.append(change_lr)
        
        if P['stop_early']:
            stop_early = EarlyStopping(monitor='val_loss',
                                    patience=P['patience'],
                                    verbose=0, mode='auto')
            callbacks.append(stop_early)
            
        logger = FileLogger(log_file_path=os.path.join(results_dir, 
                                            "training_log_" +
                                            str(experiment_ID) + ".txt"))
        callbacks.append(logger)
    
    
    best_weights_path = os.path.join(results_dir, "best_weights_" + 
                                     str(experiment_ID) + ".hdf5")
    checkpointer_best = ModelCheckpoint(filepath=best_weights_path, 
                                        verbose=1, save_best_only=True)   
    callbacks.append(checkpointer_best)
    
    last_weights_path = os.path.join(results_dir, "last_weights_" + 
                                     str(experiment_ID) + ".hdf5")
    checkpointer_last = ModelCheckpoint(filepath=last_weights_path, 
                                        verbose=0, save_best_only=False)   
    callbacks.append(checkpointer_last)
    
        
    # Train model
    if P['lr_schedule'][0]=='constant':
        history = model.fit_generator(datagen.flow(X_train, 
                                                    Y_train, 
                                                    batch_size=P['batch_size'],
                                                    shuffle=True),
                                      samples_per_epoch=len(X_train), 
                                      nb_epoch=P['nb_epoch'],
                                      callbacks=callbacks,
                                      validation_data=(X_test, Y_test),
                                      verbose=2)
    elif P['lr_schedule'][0]=='valid':
        history = {}
        lr_schedule_epochs = []
        for i, lr in enumerate(P['lr_schedule'][1]):
            # Set up callbacks
            logger = FileLogger(log_file_path=os.path.join(results_dir, 
                                                    "training_log_" +
                                                    str(experiment_ID) +
                                                    "__phase_"+str(i)+".txt"))
            stop_early = EarlyStopping(monitor='val_loss',
                                    patience=P['patience'],
                                    verbose=0, mode='auto')
            
            # Load best model weights
            if i>0:
                input = Input(shape=(3, 32, 32))
                model = assemble_model(input, **model_kwargs)
                model.load_weights(best_weights_path)
                model.compile(loss='categorical_crossentropy',
                            optimizer=optimizer, metrics=['accuracy'])
                
            # Set model learning rate
            model.optimizer.lr.set_value(float(lr))
            
            # Train model
            hist_i = model.fit_generator(datagen.flow(X_train, 
                                                    Y_train, 
                                                    batch_size=P['batch_size'],
                                                    shuffle=True),
                                         samples_per_epoch=len(X_train), 
                                         nb_epoch=P['nb_epoch'],
                                         callbacks=callbacks+[logger,
                                                              stop_early],
                                         validation_data=(X_test, Y_test),
                                         verbose=2)
            
            # Truncate the history to the last best saved model
            epochs_to_keep = hist_i.epoch[-1]-stop_early.wait
            lr_schedule_epochs.append(epochs_to_keep)
            for k, v in hist_i.history.items():
                history.setdefault(k, []).extend(v[:epochs_to_keep])
    
    
    # Plot history
    fig = plt.figure()
    keys = ['loss', 'val_loss', 'acc', 'val_acc']
    colors = ['red', 'blue', 'green', 'brown']
    if P['lr_schedule'][0]=='constant':
        history = history.history
    for key, c in zip(keys, colors):
        plt.plot(history[key], color=c, label=key)
        
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis([0, len(history['loss']), 0, 4])
    plt.xlabel('number of epochs')
    plt.savefig(os.path.join(results_dir,"history_"+str(experiment_ID)+".png"),
                bbox_inches='tight')