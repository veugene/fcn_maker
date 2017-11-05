from __future__ import (print_function,
                        division)
from keras.layers import (Activation,
                          Dropout,
                          AlphaDropout,
                          Lambda)
from keras.layers.merge import add as merge_add
from keras.layers.merge import concatenate as merge_concat
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import (Conv2D,
                                        Conv3D,
                                        Conv2DTranspose,
                                        Conv3DTranspose,
                                        MaxPooling2D,
                                        MaxPooling3D,
                                        UpSampling2D,
                                        UpSampling3D)
from keras.initializers import VarianceScaling
from keras.regularizers import l2
from keras import backend as K


"""
Wrappers around spatial layers to allow 2D or 3D, optionally.
"""
def Convolution(ndim=2, *args, **kwargs):
    layer = None
    if ndim==2:
        layer = Conv2D(*args, **kwargs)
    elif ndim==3:
        layer = Conv3D(*args, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")
    return layer

def ConvolutionTranspose(ndim=2, *args, **kwargs):
    layer = None
    if ndim==2:
        layer = Conv2DTranspose(*args, **kwargs)
    elif ndim==3:
        layer = Conv3DTranspose(*args, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")
    return layer
    
def MaxPooling(ndim=2, *args, **kwargs):
    if ndim==2:
        return MaxPooling2D(*args, **kwargs)
    elif ndim==3:
        return MaxPooling3D(*args, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")
    
def UpSampling(ndim=2,*args,  **kwargs):
    if ndim==2:
        return UpSampling2D(*args, **kwargs)
    elif ndim==3:
        return UpSampling3D(*args, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")
    
    
"""
Get keras's channel axis.
"""
def get_channel_axis(ndim=None):
    data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError("Unknown data_format {}".format(data_format))
    if data_format=='channels_first':
        channel_axis = 1
    else:
        if ndim is None:
            channel_axis = -1
        else:
            channel_axis = ndim+1
    return channel_axis
    
    
"""
Helper function to perform tensor merging.
"""
def merge(x, mode):
    if mode=='sum':
        out = merge_add(x)
    elif mode=='concat':
        channel_axis = get_channel_axis()
        out = merge_concat(x, axis=channel_axis)
    else:
        raise ValueError("Unrecognized merge mode: {}".format(mode))
    return out
    
    
"""
Return AlphaDropout if nonlinearity is 'selu', else Dropout.
"""
def get_dropout(dropout, nonlin=None):
    return AlphaDropout(dropout) if nonlin=='selu' else Dropout(dropout)
    
"""
Return a nonlinearity from the core library or return the provided function.
"""
def get_nonlinearity(nonlin):
    if isinstance(nonlin, str):
        return Activation(nonlin)
    return nonlin()


"""
Return a new instance of l2 regularizer, or return None.
"""
def _l2(decay):
    if decay is not None:
        return l2(decay)
    else:
        return None
    

"""
Add a unique identifier to a name string.
"""
def _get_unique_name(name, prefix=None):
    if prefix is not None:
        name = prefix + '_' + name
    name += '_' + str(K.get_uid(name))
    return name


"""
Helper function to subsample. Simple 2x decimation.
"""
def _subsample(x, ndim):
    channel_axis = get_channel_axis(ndim)
    data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))
    if ndim==2 and data_format=='channels_first':
        subsample_func = lambda x: x[:,:,::2,::2]
    elif ndim==2 and data_format=='channels_last':
        subsample_func = lambda x: x[:,::2,::2,:]
    elif ndim==3 and data_format=='channels_first':
        subsample_func = lambda x: x[:,:,::2,::2,::2]
    elif ndim==3 and data_format=='channels_last':
        subsample_func = lambda x: x[:,::2,::2,::2,:]
    else:
        raise ValueError('ndim must be 2 or 3')
    
    # Output shape.
    output_shape = list(x._keras_shape)
    spatial_dims = set(range(ndim+2)).difference([0, channel_axis])
    for dim in spatial_dims:
        output_shape[dim] = output_shape[dim]//2 + output_shape[dim]%2
    output_shape = tuple(output_shape[1:])
    
    # Execute subsampling in this layer
    x = Lambda(subsample_func, output_shape=output_shape)(x)
    
    return x


"""
Helper function to execute some upsampling mode.

conv_kwargs are:
filters : num filters
init : kernel_initializer
weight_decay : kernel_regularizer (l2)
"""
def _upsample(x, mode, ndim, **conv_kwargs):
    if mode=='repeat':
        x = UpSampling(size=2, ndim=ndim)(x)
    elif mode=='conv':
        x = ConvolutionTranspose(ndim=ndim, 
                                 strides=2,
                                 padding='valid',
                                 **conv_kwargs)(x)
    else:
        raise ValueError("Unrecognized upsample_mode: {}"
                            "".format(upsample_mode))
    return x


"""
Helper to build a norm -> relu -> conv block
This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
"""
def norm_nlin_conv(filters, kernel_size, subsample=False, upsample=False,
                   upsample_mode='repeat', nonlinearity='relu',
                   normalization=BatchNormalization, weight_decay=None, 
                   norm_kwargs=None, init='he_normal', ndim=2, name=None):
    if norm_kwargs is None:
        norm_kwargs = {}
    name = _get_unique_name('', name)
        
    def f(input):
        processed = input
        if normalization is not None:
            processed = normalization(name=name+"_norm",
                                      **norm_kwargs)(processed)
        processed = get_nonlinearity(nonlinearity)(processed)
        stride = 1
        if subsample:
            stride = 2
        if upsample:
            processed = _upsample(processed,
                                  mode=upsample_mode,
                                  ndim=ndim,
                                  filters=filters,
                                  kernel_size=2,
                                  kernel_initializer=init,
                                  kernel_regularizer=_l2(weight_decay),
                                  name=name+"_upconv")
        return Convolution(filters=filters,
                           kernel_size=kernel_size,
                           ndim=ndim,
                           strides=stride,
                           kernel_initializer=init,
                           padding='same',
                           name=name+"_conv",
                           kernel_regularizer=_l2(weight_decay))(processed)

    return f


"""
Adds a shortcut between input and residual block and merges them with 'sum'.
"""
def _shortcut(input, residual, subsample, upsample, upsample_mode='repeat',
              weight_decay=None, init='he_normal', ndim=2, name=None):
    name = _get_unique_name('shortcut', name)
    channel_axis = get_channel_axis(ndim)
    shortcut = input
    
    # Downsample input
    if subsample:
        shortcut = _subsample(shortcut, ndim=ndim)
        
    # Upsample input
    if upsample:
        shortcut = _upsample(shortcut,
                             mode=upsample_mode,
                             ndim=ndim,
                             filters=shortcut._keras_shape[channel_axis],
                             kernel_size=2,
                             kernel_initializer=init,
                             kernel_regularizer=_l2(weight_decay),
                             name=name+"_upconv")
        
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    equal_channels = residual._keras_shape[channel_axis] == \
                                            shortcut._keras_shape[channel_axis]
    if not equal_channels:
        shortcut = Convolution(filters=residual._keras_shape[channel_axis],
                               kernel_size=1, ndim=ndim,
                               kernel_initializer=init, padding='valid',
                               kernel_regularizer=_l2(weight_decay),
                               name=name+"_conv")(shortcut)
    
    out = merge_add([shortcut, residual])
        
    return out


"""
Identity block - do nothing except handle subsampling + upsampling.
"""
def identity_block(subsample=False, upsample=False, upsample_mode='repeat',
                   ndim=2, filters=32, kernel_size=2, init='he_normal',
                   weight_decay=0.0001, name=None):
    name = _get_unique_name('identity', name)
    def f(input):
        output = input
        if subsample:
            output = _subsample(output, ndim=ndim)
        if upsample:
            output = _upsample(output,
                               mode=upsample_mode,
                               ndim=ndim,
                               filters=filters,
                               kernel_size=2,
                               kernel_initializer=init,
                               kernel_regularizer=_l2(weight_decay),
                               name=name+"_upconv")
        return output
    return f


"""
Bottleneck architecture for > 34 layer resnet.
Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
Returns a final conv layer of filters * 4
"""
def bottleneck(filters, subsample=False, upsample=False,
               upsample_mode='repeat', skip=True, dropout=0.,
               normalization=BatchNormalization, weight_decay=None,
               norm_kwargs=None, init='he_normal', nonlinearity='relu',
               ndim=2, name=None):
    name = _get_unique_name('bottleneck', name)
    def f(input):
        output = norm_nlin_conv(filters,
                                kernel_size=1,
                                subsample=subsample,
                                normalization=normalization,
                                weight_decay=weight_decay,
                                norm_kwargs=norm_kwargs,
                                init=init,
                                nonlinearity=nonlinearity,
                                ndim=ndim,
                                name=name)(input)
        output = norm_nlin_conv(filters,
                                kernel_size=3,
                                normalization=normalization,
                                weight_decay=weight_decay,
                                norm_kwargs=norm_kwargs,
                                init=init,
                                nonlinearity=nonlinearity,
                                ndim=ndim,
                                name=name)(output)
        output = norm_nlin_conv(filters * 4,
                                kernel_size=1,
                                upsample=upsample,
                                upsample_mode=upsample_mode,
                                normalization=normalization,
                                weight_decay=weight_decay,
                                norm_kwargs=norm_kwargs,
                                init=init,
                                nonlinearity=nonlinearity,
                                ndim=ndim,
                                name=name)(output)
        if dropout > 0:
            output = get_dropout(dropout, nonlinearity)(output)
            
        if skip:
            output = _shortcut(input, output,
                               subsample=subsample, upsample=upsample,
                               upsample_mode=upsample_mode,
                               weight_decay=weight_decay, init=init,
                               ndim=ndim, name=name)
        return output

    return f


"""
Basic 3 X 3 convolution blocks.
Use for resnet with layers <= 34
Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
"""
def basic_block(filters, subsample=False, upsample=False,
                upsample_mode='repeat', skip=True, dropout=0.,
                normalization=BatchNormalization, weight_decay=None,
                norm_kwargs=None, init='he_normal', nonlinearity='relu',
                ndim=2, name=None):
    name = _get_unique_name('basic_block', name)
    def f(input):
        output = norm_nlin_conv(filters,
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
            output = Dropout(dropout)(output)
        output = norm_nlin_conv(filters,
                                kernel_size=3,
                                upsample=upsample,
                                upsample_mode=upsample_mode,
                                normalization=normalization,
                                weight_decay=weight_decay,
                                norm_kwargs=norm_kwargs,
                                init=init,
                                nonlinearity=nonlinearity,
                                ndim=ndim,
                                name=name)(output)
        
        if skip:
            output = _shortcut(input, output,
                               subsample=subsample,
                               upsample=upsample,
                               upsample_mode=upsample_mode,
                               weight_decay=weight_decay,
                               init=init,
                               ndim=ndim,
                               name=name)
        return output

    return f


"""
A single basic 3x3 convolution.
"""
def basic_block_mp(filters, subsample=False, upsample=False,
                   upsample_mode='repeat', skip=True, dropout=0.,
                   normalization=BatchNormalization, weight_decay=None,
                   norm_kwargs=None, init='he_normal', nonlinearity='relu',
                   ndim=2, name=None):
    if norm_kwargs is None:
        norm_kwargs = {}
    name = _get_unique_name('basic_block_mp', prefix=name)
    
    def f(input):
        output = input
        if normalization is not None:
            output = normalization(name=name+"_norm", **norm_kwargs)(output)
        output = get_nonlinearity(nonlinearity)(output)
        if subsample:
            output = MaxPooling(pool_size=2, ndim=ndim)(output)
        output = Convolution(filters=filters, kernel_size=3, 
                             ndim=ndim,
                             kernel_initializer=init,
                             padding='same',
                             kernel_regularizer=_l2(weight_decay),
                             name=name+"_conv")(output)
        if dropout > 0:
            output = get_dropout(dropout, nonlinearity)(output)
        if upsample:
            output = _upsample(output,
                               mode=upsample_mode,
                               ndim=ndim,
                               filters=filters,
                               kernel_size=2,
                               kernel_initializer=init,
                               kernel_regularizer=_l2(weight_decay),
                               name=name+"_upconv")
            
        if skip:
            output = _shortcut(input, output,
                               subsample=subsample, upsample=upsample,
                               upsample_mode=upsample_mode,
                               weight_decay=weight_decay, init=init,
                               ndim=ndim, name=name)
        return output
    
    return f


"""
Builds a residual block with repeating sub-blocks.
"""
def residual_block(block_function, filters, repetitions, skip=True,
                   dropout=0., subsample=False, upsample=False,
                   upsample_mode='repeat', normalization=BatchNormalization,
                   weight_decay=None, norm_kwargs=None, init='he_normal',
                   nonlinearity='relu', ndim=2, name=None):
    if repetitions<=0:
        raise ValueError("block repetitions (block depth) must be greater than "
                         "zero")
    def f(input):
        x = input
        for i in range(repetitions):
            subsample_i = subsample if i==0 else False
            upsample_i = upsample if i==repetitions-1 else False
            x = block_function(filters=filters,
                               skip=skip,
                               dropout=dropout,
                               subsample=subsample_i,
                               upsample=upsample_i,
                               upsample_mode=upsample_mode,
                               normalization=normalization,
                               norm_kwargs=norm_kwargs,
                               weight_decay=weight_decay,
                               nonlinearity=nonlinearity,
                               init=init,
                               ndim=ndim,
                               name=name)(x)
        return x

    return f 


"""
Two basic 3x3 convolutions with 2x2 conv upsampling, as in the UNet.
Subsampling, upsampling, and dropout handled as in the UNet.
"""
def unet_block(filters, subsample=False, upsample=False, upsample_mode='conv',
               halve_features_on_upsample=True, skip=False, dropout=0.,
               normalization=None, weight_decay=None, norm_kwargs=None,
               init='he_normal', nonlinearity='relu', ndim=2, name=None):
    name = _get_unique_name('unet_block', name)
    if norm_kwargs is None:
        norm_kwargs = {}
        
    # Filters can be an int or a tuple/list
    if hasattr(filters, '__len__'):
        filters_1, filters_2 = filters
    else:
        filters_1 = filters_2 = filters
        
    def f(input):
        output = input
        if subsample:
            output = MaxPooling(pool_size=2, ndim=ndim)(output)
        output = Convolution(filters=filters_1,
                             kernel_size=3,
                             ndim=ndim,
                             kernel_initializer=init,
                             padding='same',
                             kernel_regularizer=_l2(weight_decay),
                             name=name+"_conv")(output)
        output = norm_nlin_conv(filters=filters_2,
                                kernel_size=3,
                                normalization=normalization,
                                weight_decay=weight_decay,
                                norm_kwargs=norm_kwargs,
                                init=init,
                                nonlinearity=nonlinearity,
                                ndim=ndim,
                                name=name)(output)
        if normalization is not None:
            output = normalization(name=name+"_norm", **norm_kwargs)(output)
        output = get_nonlinearity(nonlinearity)(output)
        if dropout > 0:
            output = get_dropout(dropout, nonlinearity)(output)
        if upsample:
            # "up-convolution" in standard 2D unet halves the number of 
            # feature maps - but not in the standard 3D unet. It's just a
            # user-settable option in this block, regardless of ndim.
            if halve_features_on_upsample:
                filters_up = filters_2//2
            else:
                filters_up = filters_2
            output = _upsample(output,
                               mode=upsample_mode,
                               ndim=ndim,
                               filters=filters_up,
                               kernel_size=2,
                               kernel_initializer=init,
                               kernel_regularizer=_l2(weight_decay),
                               name=name+"_upconv")
            output = get_nonlinearity(nonlinearity)(output)
        if skip:
            output = _shortcut(input, output,
                               subsample=subsample,
                               upsample=upsample,
                               upsample_mode=upsample_mode,
                               weight_decay=weight_decay,
                               init=init,
                               ndim=ndim,
                               name=name)
        return output

    return f


"""
Processing block as in the VNet.
"""
def vnet_block(filters, num_conv=3, subsample=False, upsample=False,
               upsample_mode='conv', skip=True, dropout=0., normalization=None,
               norm_kwargs=None,
               init=VarianceScaling(scale=3., mode='fan_avg'),
               weight_decay=None, nonlinearity='relu', ndim=3, name=None):
    name = _get_unique_name('vnet_block', name)
    if norm_kwargs is None:
        norm_kwargs = {}
    def f(input):
        output = input
        if subsample:
            output = Convolution(filters=filters,
                                 kernel_size=2,
                                 strides=2,
                                 ndim=ndim,
                                 kernel_initializer=init,
                                 padding='same',
                                 kernel_regularizer=_l2(weight_decay),
                                 name=name+"_downconv")(output)
        for i in range(num_conv):
            output = norm_nlin_conv(filters,
                                    kernel_size=5,
                                    normalization=normalization,
                                    weight_decay=weight_decay,
                                    norm_kwargs=norm_kwargs,
                                    init=init,
                                    nonlinearity=nonlinearity,
                                    ndim=ndim,
                                    name=name)(output)
        
            if dropout > 0:
                output = get_dropout(dropout, nonlinearity)(output)
        if skip:
            output = _shortcut(input, output,
                               subsample=subsample,
                               upsample=False,
                               upsample_mode=upsample_mode,
                               weight_decay=weight_decay,
                               init=init,
                               ndim=ndim,
                               name=name)
        if upsample:
            # "up-convolution" also halves the number of feature maps.
            if normalization is not None:
                output = normalization(name=name+"_norm", **norm_kwargs)(output)
            output = get_nonlinearity(nonlinearity)(output)
            output = _upsample(output,
                               mode=upsample_mode,
                               ndim=ndim,
                               filters=filters//2,
                               kernel_size=2,
                               kernel_initializer=init,
                               kernel_regularizer=_l2(weight_decay),
                               name=name+"_upconv")
            output = get_nonlinearity(nonlinearity)(output)
        return output

    return f


"""
Dense block (as in a DenseNet), as implemented in the 100 layer Tiramisu.

paper : https://arxiv.org/abs/1611.09326 (version 2)
code  : https://github.com/SimJeg/FC-DenseNet
        commit ee933144949d82ada32198e49d76b708f60e4
"""
def dense_block(filters, block_depth=4, subsample=False, upsample=False,
                upsample_mode='conv', skip_merge_mode='concat',
                merge_input=True, dropout=0., normalization=BatchNormalization,
                norm_kwargs=None, weight_decay=None, init='he_uniform',
                nonlinearity='relu', ndim=2, name=None):
    name = _get_unique_name('dense_block', name)
    if norm_kwargs is None:
        norm_kwargs = {}        
    channel_axis = get_channel_axis(ndim)
        
    def f(input):
        output = input
        
        # Transition down (preserve num filters)
        if subsample:
            output = norm_nlin_conv(filters=output._keras_shape[channel_axis],
                                    kernel_size=1,
                                    normalization=normalization,
                                    weight_decay=weight_decay,
                                    norm_kwargs=norm_kwargs,
                                    init=init,
                                    nonlinearity=nonlinearity,
                                    ndim=ndim,
                                    name=name)(output)
            if dropout > 0:
                output = get_dropout(dropout, nonlinearity)(output)
            output = MaxPooling(pool_size=2, ndim=ndim)(output)
        
        # Book keeping.
        tensors = [output]
        
        # If 'sum' mode, make the channel dimension match.
        if skip_merge_mode=='sum':
            if tensors[0]._keras_shape[channel_axis] != filters:
                tensors[0] = Convolution(filters=filters,
                                         kernel_size=1,
                                         ndim=ndim,
                                         kernel_initializer=init,
                                         padding='valid',
                                         kernel_regularizer=_l2(weight_decay),
                                         name=name+"_adapt_conv")(tensors[0])
                
        # Build the dense block.
        for i in range(block_depth):
            output = norm_nlin_conv(filters,
                                    kernel_size=3,
                                    normalization=normalization,
                                    weight_decay=weight_decay,
                                    norm_kwargs=norm_kwargs,
                                    init=init,
                                    nonlinearity=nonlinearity,
                                    ndim=ndim,
                                    name=name)(output)
            if dropout > 0:
                output = get_dropout(dropout, nonlinearity)(output)
            tensors.append(output)
            output = merge(tensors, mode=skip_merge_mode)
        
        # Block's output - merge input in?
        #
        # Regardless, all representations inside the block (all conv outputs)
        # are merged together, forming a dense skip pattern.
        output = tensors[-1]
        if merge_input:
            # Merge the block's input into its output.
            if len(tensors) > 1:
                output = merge(tensors, mode=skip_merge_mode)
        else:
            # Avoid merging the block's input into its output.
            # With this, one can avoid exponential growth in num of filters.
            if len(tensors[1:]) > 1:
                output = merge(tensors[1:], mode=skip_merge_mode)
        
        # Transition up (maintain num filters)
        if upsample:
            output = _upsample(output,
                               mode=upsample_mode,
                               ndim=ndim,
                               filters=output._keras_shape[channel_axis],
                               kernel_size=3,
                               kernel_initializer=init,
                               kernel_regularizer=_l2(weight_decay),
                               name=name+"_upconv")
        
        return output
    
    return f
