 
import sys
sys.path.append("../")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import tifffile as tf
import os
from lib.fcn import assemble_model
from lib.loss import dice_loss
from lib.blocks import (basic_block_mp,
                        basic_block,
                        bottleneck)
from keras.preprocessing.image import ImageDataGenerator
from lib.logging import FileLogger
from keras.callbacks import (EarlyStopping, 
                             LearningRateScheduler, 
                             ModelCheckpoint)
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
import sys
import shutil
#from theano import tensor as T
from collections import OrderedDict
sys.setrecursionlimit(99999)


# -----------------------------------------------------------------------------
model_kwargs = OrderedDict((
    ('input_shape', (1, 512, 512)),
    ('num_classes', 1),
    ('input_num_filters', 16),
    ('main_block_depth', [3, 8, 10, 3]),
    ('num_main_blocks', 3),
    ('num_init_blocks', 1),
    ('weight_decay', 0.0001), 
    ('dropout', 0.1),
    ('short_skip', True),
    ('long_skip', True),
    ('long_skip_merge_mode', 'sum'),
    ('use_skip_blocks', False),
    ('relative_num_across_filters', 1),
    ('mainblock', bottleneck),
    ('initblock', basic_block_mp)
    ))
P = OrderedDict((
    ('n_train', 26),
    ('batch_size', 4),
    ('nb_epoch', 500),
    ('lr_schedule', False),
    ('early_stopping', False),
    ('initial_lr', 0.001),
    ('optimizer_type', 'RMSprop'), # 'RMSprop' or 'SGD'
    ))
training = True
experiment_ID = "example-resunet"
results_dir = "/tmp/results"

np.random.seed(1337)
# -----------------------------------------------------------------------------

data_path = "/tmp/datasets/isbi_2012_em/"
ds_path = {'train-volume': data_path + "train-volume.tif",
           'train-labels': data_path + "train-labels.tif",
           'test-volume': data_path + "test-volume.tif"}

if P['optimizer_type'] == 'RMSprop':
    optimizer = RMSprop(lr=P['initial_lr'], rho=0.9, epsilon=1e-08)
else:
    optimizer = SGD(lr=P['initial_lr'], momentum=0.9, decay=0.0, nesterov=True)

''' Learning rate scheduler '''
def scheduler(epoch):
    if epoch%200==0 and epoch>0:
        tmp_lr = (model.optimizer.lr.get_value()/10).astype('float32')
        model.optimizer.lr.set_value(tmp_lr)        
    lr = model.optimizer.lr.get_value()
    return np.float(lr)

''' Accuracy metric '''
def accuracy(y_true, y_pred):
    if model_kwargs['num_classes']==1:
        return K.mean(K.equal(y_true, K.round(y_pred)))
    else:
        return K.mean(K.equal(K.squeeze(y_true, 1),
                              K.argmax(y_pred, axis=-1)))


"""
RUN
"""

# Print settings to screen
print("Experiment:", experiment_ID)
print("")
for key in model_kwargs.keys():
    print(key, ":", model_kwargs[key])
print("")
for key in P.keys():
    print(key, ":", P[key])

if training:
    # Set up experiment directory
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
        
    print("Preparing data")

    # Load data
    X = tf.imread(ds_path['train-volume']).astype(np.float32)
    Y = tf.imread(ds_path['train-labels']).astype(np.int32)
    Y[Y==255] = 1

    # Invert labels (if dice loss)
    if model_kwargs['num_classes']==1:
        Y = 1-Y

    # Standardize the data
    mean = X.reshape(X.shape[0],
                     np.prod(X.shape[1:])).mean(axis=-1)[:,None,None]
    std  = X.reshape(X.shape[0],
                     np.prod(X.shape[1:])).std(axis=-1)[:,None,None]
    X = (X-mean)/std

    # Add dimension
    X = np.expand_dims(X, axis=1)
    Y = np.expand_dims(Y, axis=1)
    
    print("Preparing model")
    model = assemble_model(**model_kwargs)
    
    # save experiment script
    fn = sys.argv[0].rsplit('/', 1)[-1]
    shutil.copy(sys.argv[0], os.path.join(results_dir, fn))
    
    # save model in yaml
    yaml_string = model.to_yaml()
    open(os.path.join(results_dir, "model_" +
                      str(experiment_ID) +
                      ".yaml"), 'w').write(yaml_string)

    #model.summary()
    model.compile(loss=dice_loss(), 
                  optimizer=optimizer, 
                  metrics=[accuracy])

    # Permute data randomly
    R = np.random.permutation(len(X))
    X = X[R]
    Y = Y[R]

    # Split the data into train, validation
    X_train = X[:P['n_train']]
    Y_train = Y[:P['n_train']]
    X_val = X[P['n_train']:]
    Y_val = Y[P['n_train']:]

    # Prepare data augmentation
    datagen = ImageDataGenerator(rotation_range=25,
                                shear_range=0.41,
                                horizontal_flip=True,
                                vertical_flip=True,
                                fill_mode='reflect')

                                
    # Callbacks
    callbacks=[]
    checkpointer = ModelCheckpoint(filepath=os.path.join(results_dir, 
                                                        "weights_resunet_" + 
                                                        str(experiment_ID) + 
                                                        ".hdf5"), 
                                   verbose=1, save_best_only=True)
    callbacks.append(checkpointer)
    logger = FileLogger(log_file_path=os.path.join(results_dir, 
                                                  "training_log_" +
                                                  str(experiment_ID) + ".txt"))
    callbacks.append(logger)
    if P['lr_schedule']:
        change_lr = LearningRateScheduler(scheduler)
        callbacks.append(change_lr)
    if P['early_stopping']:
        early_stopping = EarlyStopping(monitor='val_loss', 
                                       patience=200, verbose=0, mode='auto')
        callbacks.append(early_stopping)
        
    # Run fit
    print("Training")
    history = model.fit_generator(datagen.flow(X_train, 
                                               Y_train, 
                                               batch_size=P['batch_size'], 
                                               shuffle=True),
                                  samples_per_epoch=len(X_train), 
                                  nb_epoch=P['nb_epoch'],
                                  callbacks=callbacks,
                                  validation_data=(X_val, Y_val),
                                  verbose=2)
    
    # Plot history
    fig = plt.figure()
    keys = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
    colors = ['red', 'blue', 'green', 'brown']
    for key, c in zip(keys, colors):
        plt.plot(history.history[key], color=c, label=key)
       
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis([0, P['nb_epoch'], -1, 1])
    plt.xlabel('number of epochs')
    plt.savefig(os.path.join(results_dir,
                             "history_" +
                             str(experiment_ID) +
                             ".png"), bbox_inches='tight')
   
else:
    # Prediction
    X = tf.imread(ds_path['test-volume']).astype(np.float32)
    mean = X.reshape(X.shape[0],
                     np.prod(X.shape[1:])).mean(axis=-1)[:,None,None]
    std  = X.reshape(X.shape[0],
                     np.prod(X.shape[1:])).std(axis=-1)[:,None,None]
    X = (X-mean)/std
    X = np.expand_dims(X, axis=1)

    model = assemble_model(**model_kwargs)
    model.load_weights(os.path.join(results_dir, "weights_resunet_" + 
                                    str(experiment_ID) + ".hdf5"))
                       
    model.compile(loss=dice_loss(), 
                  optimizer=optimizer, 
                  metrics=[accuracy])                                
    
    X_pred = np.squeeze(model.predict(X, batch_size=P['batch_size']))
    X_pred = np.transpose(X_pred, (0, 2, 1))
    
    tf.imsave(os.path.join(results_dir, 
                           "pred_train_val_" +
                           str(experiment_ID) +
                           ".tif"),
              X_pred.astype('float32'))       
                                    
