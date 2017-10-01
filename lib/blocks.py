from keras.layers import (Activation,
                          Dropout,
                          AlphaDropout,
                          Lambda)
from keras.layers.merge import add as merge_add
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import (Convolution2D,
                                        Convolution3D,
                                        MaxPooling2D,
                                        MaxPooling3D,
                                        UpSampling2D,
                                        UpSampling3D)
from keras.regularizers import l2
from keras import backend as K


"""
Wrappers around spatial layers to allow 2D or 3D, optionally.
"""
def Convolution(*args, ndim=2, **kwargs):
    layer = None
    if ndim==2:
        layer = Convolution2D(*args, **kwargs)
    elif ndim==3:
        layer = Convolution3D(*args, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")
    return layer
    
def MaxPooling(*args, ndim=2, **kwargs):
    if ndim==2:
        return MaxPooling2D(*args, **kwargs)
    elif ndim==3:
        return MaxPooling3D(*args, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")
    
def UpSampling(*args, ndim=2, **kwargs):
    if ndim==2:
        return UpSampling2D(*args, **kwargs)
    elif ndim==3:
        return UpSampling3D(*args, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")
    
    
"""
Return a nonlinearity from the core library or return the provided function.
"""
def get_nonlinearity(nonlin):
    if isinstance(nonlin, str):
        return Activation(nonlin)
    return nonlin()


# Return a new instance of l2 regularizer, or return None
def _l2(decay):
    if decay is not None:
        return l2(decay)
    else:
        return None
    

# Add a unique identifier to a name string.
def _get_unique_name(name, prefix=None):
    if prefix is not None:
        name = prefix + '_' + name
    name += '_' + str(K.get_uid(name))
    return name


# Helper to build a norm -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _norm_relu_conv(filters, kernel_size, subsample=False, upsample=False,
                    nonlinearity='relu', normalization=BatchNormalization,
                    weight_decay=None,  norm_kwargs=None, init='he_normal',
                    ndim=2, name=None):
    if norm_kwargs is None:
        norm_kwargs = {}
    name = _get_unique_name('', name)
        
    def f(input):
        processed = input
        if normalization is not None:
            processed = normalization(name=name+'_norm',
                                      **norm_kwargs)(processed)
        processed = get_nonlinearity(nonlinearity)(processed)
        stride = 1
        if subsample:
            stride = 2
        if upsample:
            processed = UpSampling(size=2, ndim=ndim)(processed)
        return Convolution(filters=filters, kernel_size=kernel_size, ndim=ndim,
                           strides=stride,
                           kernel_initializer=init,
                           padding='same', name=name+'_conv',
                           kernel_regularizer=_l2(weight_decay))(processed)

    return f


# Adds a shortcut between input and residual block and merges them with 'sum'
def _shortcut(input, residual, subsample, upsample, normalization=None,
              weight_decay=None, init='he_normal', ndim=2, name=None):
    name = _get_unique_name('shortcut', name)
    
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
        if ndim==2:
            shortcut = Lambda(lambda x: x[:,:, ::2, ::2],
                              output_shape=downsample_output_shape)(shortcut)
        elif ndim==3:
            shortcut = Lambda(lambda x: x[:,:,:, ::2, ::2],
                              output_shape=downsample_output_shape)(shortcut)
        else:
            raise ValueError("ndim must be 2 or 3")
        
    # Upsample input
    if upsample:
        shortcut = UpSampling(size=2, ndim=ndim)(shortcut)
        
    # Adjust input channels to match residual
    if not equal_channels:
        shortcut = Convolution(filters=residual._keras_shape[1],
                               kernel_size=1, ndim=ndim,
                               kernel_initializer=init, padding='valid',
                               kernel_regularizer=_l2(weight_decay),
                               name=name+'_conv')(shortcut)
    
    out = merge_add([shortcut, residual])
        
    return out


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of filters * 4
def bottleneck(filters, subsample=False, upsample=False, skip=True,
               dropout=0., normalization=BatchNormalization, weight_decay=None,
               num_residuals=1, norm_kwargs=None, init='he_normal', 
               nonlinearity='relu', ndim=2, name=None):
    name = _get_unique_name('bottleneck', name)
    def f(input):
        residuals = []
        for i in range(num_residuals):
            residual = _norm_relu_conv(filters,
                                       kernel_size=1,
                                       subsample=subsample,
                                       normalization=normalization,
                                       weight_decay=weight_decay,
                                       norm_kwargs=norm_kwargs,
                                       init=init,
                                       nonlinearity=nonlinearity,
                                       ndim=ndim,
                                       name=name)(input)
            residual = _norm_relu_conv(filters,
                                       kernel_size=3,
                                       normalization=normalization,
                                       weight_decay=weight_decay,
                                       norm_kwargs=norm_kwargs,
                                       init=init,
                                       nonlinearity=nonlinearity,
                                       ndim=ndim,
                                       name=name)(residual)
            residual = _norm_relu_conv(filters * 4,
                                       kernel_size=1,
                                       upsample=upsample,
                                       normalization=normalization,
                                       weight_decay=weight_decay,
                                       norm_kwargs=norm_kwargs,
                                       init=init,
                                       nonlinearity=nonlinearity,
                                       ndim=ndim,
                                       name=name)(residual)
            if dropout > 0:
                if nonlinearity=='selu':
                    residual = AlphaDropout(dropout)(residual)
                else:
                    residual = Dropout(dropout)(residual)
            residuals.append(residual)
            
        if len(residuals)>1:
            output = merge_add(residuals)
        else:
            output = residuals[0]
        if skip:
            output = _shortcut(input, output,
                               subsample=subsample, upsample=upsample,
                               normalization=normalization,
                               weight_decay=weight_decay, init=init,
                               ndim=ndim, name=name)
        return output

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def basic_block(filters, subsample=False, upsample=False, skip=True,
                dropout=0., normalization=BatchNormalization, 
                weight_decay=None, num_residuals=1, norm_kwargs=None,
                init='he_normal', nonlinearity='relu', ndim=2, name=None):
    name = _get_unique_name('basic_block', name)
    def f(input):
        residuals = []
        for i in range(num_residuals):
            residual = _norm_relu_conv(filters,
                                       kernel_size=3,
                                       subsample=subsample,
                                       normalization=normalization,
                                       weight_decay=weight_decay,
                                       norm_kwargs=norm_kwargs,
                                       init=init,
                                       nonlinearity=nonlinearity,
                                       ndim=ndim,
                                       name=name)(input)
            if dropout > 0:
                residual = Dropout(dropout)(residual)
            residual = _norm_relu_conv(filters,
                                       kernel_size=3,
                                       upsample=upsample,
                                       normalization=normalization,
                                       weight_decay=weight_decay,
                                       norm_kwargs=norm_kwargs,
                                       init=init,
                                       nonlinearity=nonlinearity,
                                       ndim=ndim,
                                       name=name)(residual)
            residuals.append(residual)
        
        if len(residuals)>1:
            output = merge_add(residuals)
        else:
            output = residuals[0]
        if skip:
            output = _shortcut(input, output,
                               subsample=subsample,
                               upsample=upsample,
                               normalization=normalization,
                               weight_decay=weight_decay,
                               init=init,
                               ndim=ndim,
                               name=name)
        return output

    return f


# Builds a residual block with repeating bottleneck blocks.
def residual_block(block_function, filters, repetitions, num_residuals=1,
                   skip=True, dropout=0., subsample=False, upsample=False,
                   normalization=BatchNormalization, weight_decay=None,
                   norm_kwargs=None, init='he_normal', nonlinearity='relu',
                   ndim=2, name=None):
    def f(input):
        x = input
        for i in range(repetitions):
            subsample_i = subsample if i==0 else False
            upsample_i = upsample if i==repetitions-1 else False
            x = block_function(filters=filters,
                                   num_residuals=num_residuals,
                                   skip=skip,
                                   dropout=dropout,
                                   subsample=subsample_i,
                                   upsample=upsample_i,
                                   normalization=normalization,
                                   norm_kwargs=norm_kwargs,
                                   weight_decay=weight_decay,
                                   nonlinearity=nonlinearity,
                                   init=init,
                                   ndim=ndim,
                                   name=name)(x)
        return x

    return f 


# A single basic 3x3 convolution
def basic_block_mp(filters, subsample=False, upsample=False, skip=True,
                   dropout=0., normalization=BatchNormalization,
                   weight_decay=None, num_residuals=1, norm_kwargs=None,
                   init='he_normal', nonlinearity='relu', ndim=2, name=None):
    if norm_kwargs is None:
        norm_kwargs = {}
    name = _get_unique_name('basic_block_mp', prefix=name)
    
    def f(input):
        residuals = []
        for i in range(num_residuals):
            residual = input
            if normalization is not None:
                residual = normalization(name=name+"_norm_"+str(i),
                                         **norm_kwargs)(residual)
            residual = get_nonlinearity(nonlinearity)(residual)
            if subsample:
                residual = MaxPooling(pool_size=2, ndim=ndim)(residual)
            residual = Convolution(filters=filters, kernel_size=3, 
                                   ndim=ndim,
                                   kernel_initializer=init,
                                   padding='same',
                                   kernel_regularizer=_l2(weight_decay),
                                   name=name+"_conv_"+str(i))(residual)
            if dropout > 0:
                if nonlinearity=='selu':
                    residual = AlphaDropout(dropout)(residual)
                else:
                    residual = Dropout(dropout)(residual)
            if upsample:
                residual = UpSampling(size=2, ndim=ndim)(residual)
            residuals.append(residual)
            
        if len(residuals)>1:
            output = merge_add(residuals)
        else:
            output = residuals[0]
        if skip:
            output = _shortcut(input, output,
                               subsample=subsample, upsample=upsample,
                               normalization=normalization,
                               weight_decay=weight_decay, init=init,
                               ndim=ndim, name=name)
        return output
    
    return f
