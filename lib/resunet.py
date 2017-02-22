from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Dense,
                          Permute,
                          Lambda,
                          merge)
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras import backend as K
from theano import tensor as T
from keras.regularizers import l2
import numpy as np
from .blocks import (bottleneck,
                     basic_block,
                     basic_block_mp,
                     residual_block)
from .loss import (categorical_crossentropy_ND,
                   dice_loss,
                   masked_dice_loss,
                   cce_with_regional_penalty)


def _l2(decay):
    """
    Return a new instance of l2 regularizer, or return None
    """
    if decay is not None:
        return l2(decay)
    else:
        return None
    

def _softmax(x):
    """
    Softmax that works on ND inputs.
    """
    e = K.exp(x - K.max(x, axis=-1, keepdims=True))
    s = K.sum(e, axis=-1, keepdims=True)
    return e / s

    
class _layer_tracker(object):
    """
    Helper object to keep track of previously added layer and allow layer
    retrieval by name from a dictionary.
    """
    def __init__(self):
        self.layer_dict = {}
        self.prev_layer = None
        
    def record(self, layer, name):
        layer.name = name
        self.layer_dict[name] = layer
        self.prev_layer = layer
        
    def __getitem__(self, name):
        return self.layer_dict[name]
    
    
#def _pad_to_fit(x, target_shape):
    #"""
    #Spatially pad a tensor's feature maps with zeros as evenly as possible
    #(center it) to fit the target shape.
    
    #Expected target shape is larger than the shape of the tensor.
    
    #NOTE: padding may be unequal on either side of the map if the target
    #dimension is odd. This is why keras's ZeroPadding2D isn't used.
    #"""
    #pad_0 = {}
    #pad_1 = {}
    #for dim in [2, 3]:
        #pad_0[dim] = (target_shape[dim]-x.shape[dim])//2
        #pad_1[dim] = target_shape[dim]-x.shape[dim]-pad_0[dim]
    #output = T.zeros(target_shape)
    #indices = (slice(None),
               #slice(None),
               #slice(pad_0[2], target_shape[2]-pad_1[2]),
               #slice(pad_0[3], target_shape[3]-pad_1[3]))
    #return T.set_subtensor(output[indices], x)
    
    
def _make_long_skip(prev_layer, concat_layer, num_concat_filters, bn_kwargs,
                    num_target_filters, use_skip_blocks, repetitions,
                    dropout, skip, batch_norm, weight_decay, num_residuals,
                    merge_mode='concat', block=bottleneck):
    """
    Helper function to create a long skip connection with concatenation.
    Concatenated information is not transformed if use_skip_blocks is False.
    """
    if use_skip_blocks:
        concat_layer = residual_block(block, nb_filter=num_concat_filters,
                           repetitions=repetitions, dropout=dropout, skip=skip,
                           batch_norm=batch_norm, bn_kwargs=bn_kwargs,
                           weight_decay=weight_decay)(concat_layer)
    if merge_mode == 'sum':
        if prev_layer._keras_shape[1] != num_target_filters:
            prev_layer = Convolution2D(num_target_filters, 1, 1,
                                 init='he_normal', border_mode='valid',
                                 W_regularizer=_l2(weight_decay))(prev_layer)
        if concat_layer._keras_shape[1] != num_target_filters:
            concat_layer = Convolution2D(num_target_filters, 1, 1,
                                 init='he_normal', border_mode='valid',
                                 W_regularizer=_l2(weight_decay))(concat_layer)
    #zero_pad = Lambda(_pad_to_fit,
                      #output_shape=concat_layer._keras_shape[1:],
                      #arguments={'target_shape': concat_layer.shape})
    #prev_layer = zero_pad(prev_layer)
    merged = merge([prev_layer, concat_layer], mode=merge_mode, concat_axis=1)
    return merged
    
    
def assemble_model(input_shape, num_classes, num_main_blocks, main_block_depth,
                   num_init_blocks, input_num_filters, short_skip=True,
                   long_skip=True, long_skip_merge_mode='concat',
                   mainblock=None, initblock=None, use_skip_blocks=True,
                   skipblock=None, relative_num_across_filters=1,
                   num_residuals=1, dropout=0., batch_norm=True,
                   weight_decay=None, bn_kwargs=None):
    """
    input_shape : tuple specifiying the 2D image input shape.
    num_classes : number of classes in the segmentation output.
    num_main_blocks : the number of blocks of type mainblock, below initblocks.
        These blocks double (halve) in number of channels at each downsampling
        (upsampling).
    main_block_depth : an integer or list of integers specifying the number of
        repetitions of each mainblock. A list must contain as many values as
        there are main_blocks in the downward (or upward -- it's mirrored) path
        plus one for the across path.
    num_init_blocks : the number of blocks of type initblock, above mainblocks.
        These blocks always have the same number of channels as the first
        convolutional layer in the model.
    input_num_filters : the number channels in the first (last) convolutional
        layer in the model (and of each initblock).
    short_skip : ResNet-like shortcut connections from the input of each block
        to its output. The inputs are summed with the outputs.
    long_skip : UNet-like skip connections from the downward path to the upward
        path. These can either concatenate or sum features across.
    long_skip_merge_mode : Either 'concat' or 'sum' features across long_skip.
    mainblock : a layer defining the mainblock (bottleneck by default).
    initblock : a layer defining the initblock (basic_block_mp by default).
    use_skip_blocks : pass features skipped along long_skip through skipblocks.
    skipblock : a layer defining the skipblock (basic_block_mp by default).
    relative_num_across_filters : multiply the number of channels in the across
        path (and in each skipblock, if they exist) by this value.
    num_residuals : the number of parallel residual functions per block.
    dropout : the dropout probability, introduced in every block.
    batch_norm : enable or disable batch normalization.
    weight_decay : the weight decay (L2 penalty) used in every convolution.
    bn_kwargs : keyword arguments for keras batch normalization.
    """
    
    '''
    By default, use depth 2 bottleneck for mainblock
    '''
    if mainblock is None:
        mainblock = bottleneck
    if initblock is None:
        initblock = basic_block_mp
    if skipblock is None:
        skipblock = basic_block_mp
    
    '''
    main_block_depth can be a list per block or a single value 
    -- ensure the list length is correct (if list) and that no length is 0
    '''
    if not hasattr(main_block_depth, '__len__'):
        if main_block_depth==0:
            raise ValueError("main_block_depth must never be zero")
    else:
        if len(main_block_depth)!=num_main_blocks+1:
            raise ValueError("main_block_depth must have " 
                             "`num_main_blocks+1` values when " 
                             "passed as a list")
        for d in main_block_depth:
            if d==0:
                raise ValueError("main_block_depth must never be zero")
    
    '''
    Returns the depth of a mainblock for a given pooling level
    '''
    def get_repetitions(level):
        if hasattr(main_block_depth, '__len__'):
            return main_block_depth[level]
        return main_block_depth
    
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'skip': short_skip,
                    'dropout': dropout,
                    'batch_norm': batch_norm,
                    'weight_decay': weight_decay,
                    'num_residuals': num_residuals,
                    'bn_kwargs': bn_kwargs}
    
    '''
    If long skip is not (the defualt) identity, always pass these
    parameters to _make_long_skip
    '''
    long_skip_kwargs = {'use_skip_blocks': use_skip_blocks,
                        'repetitions': 1,
                        'merge_mode': long_skip_merge_mode,
                        'block': skipblock}
    long_skip_kwargs.update(block_kwargs)
    
    layers = _layer_tracker()
    
    # INPUT
    input = Input(shape=input_shape)
    
    # Initial convolution
    layers.record(Convolution2D(input_num_filters, 3, 3,
                               init='he_normal', border_mode='same',
                               W_regularizer=_l2(weight_decay))(input),
                  name='first_conv')
    
    # DOWN (initial subsampling blocks)
    for b in range(num_init_blocks):
        layers.record(initblock(input_num_filters, subsample=True,
                                **block_kwargs)(layers.prev_layer),
                      name='initblock_d'+str(b))
        print("INIT DOWN {}: {} -- {}".format(b, layers.prev_layer.name,
                                              layers.prev_layer._keras_shape))
    
    # DOWN (resnet blocks)
    for b in range(num_main_blocks):
        num_filters = input_num_filters*(2**b)
        layers.record(residual_block(mainblock, nb_filter=num_filters, 
                              repetitions=get_repetitions(b), subsample=True,
                              **block_kwargs)(layers.prev_layer),
                      name='mainblock_d'+str(b))
        print("MAIN DOWN {}: {} (depth {}) -- {}".format(b,
              layers.prev_layer.name, get_repetitions(b),
              layers.prev_layer._keras_shape))
        
    # ACROSS
    num_filters = input_num_filters*(2**num_main_blocks)
    num_filters *= relative_num_across_filters
    layers.record(residual_block(mainblock, nb_filter=num_filters, 
                                 repetitions=get_repetitions(num_main_blocks),
                                 subsample=True, upsample=True,
                                 **block_kwargs)(layers.prev_layer), 
                  name='mainblock_a')
    print("ACROSS: {} (depth {}) -- {}".format( \
          layers.prev_layer.name, get_repetitions(num_main_blocks),
          layers.prev_layer._keras_shape))

    # UP (resnet blocks)
    for b in range(num_main_blocks-1, -1, -1):
        num_filters = input_num_filters*(2**b)
        if long_skip:
            num_across_filters = num_filters*relative_num_across_filters
            repetitions = get_repetitions(num_main_blocks)
            layers.record(_make_long_skip(prev_layer=layers.prev_layer,
                                     concat_layer=layers['mainblock_d'+str(b)],
                                     num_concat_filters=num_across_filters,
                                     num_target_filters=num_filters,
                                     **long_skip_kwargs),
                          name='concat_main_'+str(b))
        layers.record(residual_block(mainblock, nb_filter=num_filters, 
                               repetitions=get_repetitions(b), upsample=True,
                               **block_kwargs)(layers.prev_layer),
                      name='mainblock_u'+str(b))
        print("MAIN UP {}: {} (depth {}) -- {}".format(b,
              layers.prev_layer.name, get_repetitions(b),
              layers.prev_layer._keras_shape))
        
    # UP (final upsampling blocks)
    for b in range(num_init_blocks-1, -1, -1):
        if long_skip:
            num_across_filters = input_num_filters*relative_num_across_filters
            repetitions = get_repetitions(num_main_blocks)
            layers.record(_make_long_skip(prev_layer=layers.prev_layer,
                                     concat_layer=layers['initblock_d'+str(b)],
                                     num_concat_filters=num_across_filters,
                                     num_target_filters=input_num_filters,
                                     **long_skip_kwargs),
                          name='concat_init_'+str(b))  
        layers.record(initblock(input_num_filters, upsample=True,
                                **block_kwargs)(layers.prev_layer),
                      name='initblock_u'+str(b))
        print("INIT UP {}: {} -- {}".format(b,
              layers.prev_layer.name, layers.prev_layer._keras_shape))
        
    # Final convolution
    layers.record(Convolution2D(input_num_filters, 3, 3,
                               init='he_normal', border_mode='same',
                               W_regularizer=_l2(weight_decay))(layers.prev_layer),
                  name='final_conv')
    if long_skip:
        num_across_filters = input_num_filters*relative_num_across_filters
        repetitions = get_repetitions(num_main_blocks)
        layers.record(_make_long_skip(prev_layer=layers.prev_layer,
                                     concat_layer=layers['first_conv'],
                                     num_concat_filters=num_across_filters,
                                     num_target_filters=input_num_filters,
                                     **long_skip_kwargs),
                      name='concat_top')
    if batch_norm:
        if bn_kwargs is None:
            bn_kwargs = {}
        layers.record(BatchNormalization(axis=1,
                                         **bn_kwargs)(layers.prev_layer),
                      name='final_bn')
    layers.record(Activation('relu')(layers.prev_layer), name='final_relu')
    
    # OUTPUT (SOFTMAX)
    if num_classes is not None:
        # Linear classifier
        layers.record(Convolution2D(num_classes,1,1,activation='linear', 
                  W_regularizer=_l2(weight_decay))(layers.prev_layer), name='sm_1')
        layers.record(Permute((2,3,1))(layers.prev_layer), name='sm_2')
        if num_classes==1:
            output = Activation('sigmoid')(layers.prev_layer)
        else:
            output = Activation(_softmax)(layers.prev_layer)
        output = Permute((3,1,2))(output)
    else:
        # No classifier
        output = layers.prev_layer
    
    # MODEL
    model = Model(input=input, output=output)

    return model
