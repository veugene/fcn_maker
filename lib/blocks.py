from keras.layers import (Activation,
                          merge,
                          Dropout,
                          Lambda)
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import (Convolution2D,
                                        MaxPooling2D,
                                        UpSampling2D)
from keras.regularizers import l2
from keras import backend as K


# Return a new instance of l2 regularizer, or return None
def _l2(decay):
    if decay is not None:
        return l2(decay)
    else:
        return None


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=False, upsample=False,
                  batch_norm=True, weight_decay=None, bn_kwargs=None):
    if bn_kwargs is None:
        bn_kwargs = {}
        
    def f(input):
        processed = input
        if batch_norm:
            processed = BatchNormalization(axis=1, **bn_kwargs)(processed)
        processed = Activation('relu')(processed)
        stride = (1, 1)
        if subsample:
            stride = (2, 2)
        if upsample:
            processed = UpSampling2D(size=(2, 2))(processed)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
                             subsample=stride, init='he_normal',
                             border_mode='same',
                             W_regularizer=_l2(weight_decay))(processed)

    return f


# Adds a shortcut between input and residual block and merges them with 'sum'
def _shortcut(input, residual, subsample, upsample, weight_decay=None):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]
    
    shortcut = input
    
    # Downsample input
    if subsample:
        def downsample_output_shape(input_shape):
            output_shape = list(input_shape)
            output_shape[-2] = None if output_shape[-2]==None \
                                    else output_shape[-2]//2
            output_shape[-1] = None if output_shape[-1]==None \
                                    else output_shape[-1]//2
            return tuple(output_shape)
        shortcut = Lambda(lambda x: x[:,:, ::2, ::2],
                          output_shape=downsample_output_shape)(shortcut)
        
    # Upsample input
    if upsample:
        shortcut = UpSampling2D(size=(2, 2))(shortcut)
        
    # Adjust input channels to match residual
    if not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1],
                                 nb_row=1, nb_col=1,
                                 init='he_normal', border_mode='valid',
                                 W_regularizer=_l2(weight_decay))(shortcut)
        
    return merge([shortcut, residual], mode='sum')


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filter * 4
def bottleneck(nb_filter, subsample=False, upsample=False, skip=True,
               dropout=0., batch_norm=True, weight_decay=None,
               num_residuals=1, bn_kwargs=None):
    def f(input):
        residuals = []
        for i in range(num_residuals):
            residual = _bn_relu_conv(nb_filter, 1, 1,
                                      subsample=subsample,
                                      batch_norm=batch_norm,
                                      weight_decay=weight_decay,
                                      bn_kwargs=bn_kwargs)(input)
            residual = _bn_relu_conv(nb_filter, 3, 3,
                                      batch_norm=batch_norm,
                                      weight_decay=weight_decay,
                                      bn_kwargs=bn_kwargs)(residual)
            residual = _bn_relu_conv(nb_filter * 4, 1, 1,
                                      upsample=upsample,
                                      batch_norm=batch_norm,
                                      weight_decay=weight_decay,
                                      bn_kwargs=bn_kwargs)(residual)
            if dropout > 0:
                residual = Dropout(dropout)(residual)
            residiuals.append(residual)
            
        if len(residuals)>1:
            output = merge(residuals, mode='sum')
        else:
            output = residuals[0]
        if skip:
            output = _shortcut(input, output,
                               subsample=subsample, upsample=upsample,
                               weight_decay=weight_decay)
        return output

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def basic_block(nb_filter, subsample=False, upsample=False, skip=True,
                dropout=0., batch_norm=True, weight_decay=None,
                num_residuals=1, bn_kwargs=None):
    def f(input):
        residuals = []
        for i in range(num_residuals):
            residual = _bn_relu_conv(nb_filter, 3, 3,
                                     subsample=subsample,
                                     batch_norm=batch_norm,
                                     weight_decay=weight_decay,
                                     bn_kwargs=bn_kwargs)(input)
            if dropout > 0:
                residual = Dropout(dropout)(residual)
            residual = _bn_relu_conv(nb_filter, 3, 3,
                                     upsample=upsample,
                                     batch_norm=batch_norm,
                                     weight_decay=weight_decay,
                                     bn_kwargs=bn_kwargs)(residual)
            residuals.append(residual)
        
        if len(residuals)>1:
            output = merge(residuals, mode='sum')
        else:
            output = residuals[0]
        if skip:
            output = _shortcut(input, output,
                               subsample=subsample,
                               upsample=upsample,
                               weight_decay=weight_decay)
        return output

    return f


# Builds a residual block with repeating bottleneck blocks.
def residual_block(block_function, nb_filter, repetitions, num_residuals=1,
                   skip=True, dropout=0., subsample=False, upsample=False,
                   batch_norm=True, weight_decay=None, bn_kwargs=None):
    def f(input):
        for i in range(repetitions):
            kwargs = {'nb_filter': nb_filter, 'num_residuals': num_residuals,
                      'skip': skip, 'dropout': dropout, 'subsample': False,
                      'upsample': False, 'batch_norm': batch_norm,
                      'weight_decay': weight_decay, 'bn_kwargs': bn_kwargs}
            if i==0:
                kwargs['subsample'] = subsample
            if i==repetitions-1:
                kwargs['upsample'] = upsample
            input = block_function(**kwargs)(input)
        return input

    return f 


# A single basic 3x3 convolution
def basic_block_mp(nb_filter, subsample=False, upsample=False, skip=True,
                   dropout=0., batch_norm=True, weight_decay=None,
                   num_residuals=1, bn_kwargs=None):
    if bn_kwargs is None:
        bn_kwargs = {}
        
    def f(input):
        residuals = []
        for i in range(num_residuals):
            residual = input
            if batch_norm:
                residual = BatchNormalization(axis=1, **bn_kwargs)(residual)
            residual = Activation('relu')(residual)
            if subsample:
                residual = MaxPooling2D(pool_size=(2,2))(residual)
            residual = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3,
                                    init='he_normal', border_mode='same',
                                    W_regularizer=_l2(weight_decay))(residual)
            if dropout > 0:
                residual = Dropout(dropout)(residual)
            if upsample:
                residual = UpSampling2D(size=(2, 2))(residual)
            residuals.append(residual)
            
        if len(residuals)>1:
            output = merge(residuals, mode='sum')
        else:
            output = residuals[0]
        if skip:
            output = _shortcut(input, output,
                               subsample=subsample, upsample=upsample,
                               weight_decay=weight_decay)
        return output
    
    return f
