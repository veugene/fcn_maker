from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          Dense,
                          Permute,
                          Lambda,
                          add,
                          concatenate)
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from theano import tensor as T
from keras.regularizers import l2
import numpy as np
from .blocks import (bottleneck,
                     basic_block,
                     basic_block_mp,
                     residual_block,
                     Convolution)


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
    
    
def assemble_model(input_shape, num_classes, num_init_blocks, num_main_blocks,
                   main_block_depth, input_num_filters, short_skip=True,
                   long_skip=True, long_skip_merge_mode='concat',
                   mainblock=None, initblock=None, use_skip_blocks=True,
                   skipblock=None, relative_num_across_filters=1,
                   num_residuals=1, dropout=0., weight_decay=None, 
                   init='he_normal', batch_norm=True,  bn_kwargs=None, ndim=2,
                   verbose=True):
    """
    input_shape : A tuple specifiying the 2D image input shape.
    num_classes : The number of classes in the segmentation output.
    num_init_blocks : The number of blocks of type initblock, above mainblocks.
        These blocks always have the same number of channels as the first
        convolutional layer in the model.
    num_main_blocks : The number of blocks of type mainblock, below initblocks.
        These blocks double (halve) in number of channels at each downsampling
        (upsampling).
    main_block_depth : An integer or list of integers specifying the number of
        repetitions of each mainblock. A list must contain 2*num_main_blocks+1
        values (there are num_mainblocks on the contracting path and on the 
        expanding path, as well as as one on the across path). Zero is a valid
        depth.
    input_num_filters : The number channels in the first (last) convolutional
        layer in the model (and of each initblock).
    short_skip : A boolean specifying whether to use ResNet-like shortcut
        connections from the input of each block to its output. The inputs are
        summed with the outputs.
    long_skip : A boolean specifying whether to use UNet-like skip connections
        from the downward path to the upward path. These can either concatenate
        or sum features across.
    long_skip_merge_mode : Either 'concat' or 'sum' features across long_skip.
    mainblock : A layer defining the mainblock (bottleneck by default).
    initblock : A layer defining the initblock (basic_block_mp by default).
    use_skip_blocks : A boolean specifying whether to pass features skipped
        along long_skip through skipblocks.
    skipblock : A layer defining the skipblock (basic_block_mp by default).
    relative_num_across_filters : Multiply the number of channels in the across
        path (and in each skipblock, if they exist) by this integer value.
    num_residuals : The number of parallel residual functions per block.
    dropout : A float [0, 1] specifying the dropout probability, introduced in
        every block.
    weight_decay : The weight decay (L2 penalty) used in every convolution 
        (float).
    init : A string specifying (or a function defining) the initializer for
        layers.
    batch_norm : A boolean to enable or disable batch normalization.
    bn_kwargs : Keyword arguments for keras batch normalization.
    num_outputs : The number of model outputs, each with num_classifier
        classifiers.
    ndim : The spatial dimensionality of the input and output (either 2 or 3).
    verbose : A boolean specifying whether to print messages about model   
        structure during construction (if True).
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
    if hasattr(main_block_depth, '__len__'):
        if len(main_block_depth)!=2*num_main_blocks+1:
            raise ValueError("main_block_depth must have " 
                             "`2*num_main_blocks+1` values when " 
                             "passed as a list")
        
    '''
    ndim must be only 2 or 3.
    '''
    if ndim not in [2, 3]:
        raise ValueError("ndim must be either 2 or 3")
            
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'skip': short_skip,
                    'dropout': dropout,
                    'batch_norm': batch_norm,
                    'weight_decay': weight_decay,
                    'num_residuals': num_residuals,
                    'bn_kwargs': bn_kwargs,
                    'init': init,
                    'ndim': ndim}
    
    '''
    If long skip is not (the defualt) identity, always pass these
    parameters to make_long_skip
    '''
    long_skip_kwargs = {'use_skip_blocks': use_skip_blocks,
                        'repetitions': 1,
                        'merge_mode': long_skip_merge_mode,
                        'block': skipblock}
    long_skip_kwargs.update(block_kwargs)
    
    '''
    Returns the depth of a mainblock for a given pooling level
    '''
    def get_repetitions(level):
        if hasattr(main_block_depth, '__len__'):
            return main_block_depth[level]
        return main_block_depth
    
    '''
    Function to print if verbose==True
    '''
    def v_print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
        else:
            return None
    
    '''
    Helper function to create a long skip connection with concatenation.
    Concatenated information is not transformed if use_skip_blocks is False.
    '''
    def make_long_skip(prev_x, concat_x, num_concat_filters, bn_kwargs,
                       num_target_filters, use_skip_blocks, repetitions,
                       dropout, skip, batch_norm, weight_decay, num_residuals,
                       merge_mode='concat', block=bottleneck):
    
        if use_skip_blocks:
            concat_x = residual_block( \
                                       block,
                                       nb_filter=num_concat_filters,
                                       repetitions=repetitions,
                                       dropout=dropout,
                                       skip=skip,
                                       batch_norm=batch_norm,
                                       bn_kwargs=bn_kwargs,
                                       weight_decay=weight_decay)(concat_x)
        if merge_mode == 'sum':
            if prev_x._keras_shape[1] != num_target_filters:
                prev_x = Convolution( \
                                filters=num_target_filters,
                                kernel_size=1,
                                ndim=ndim,
                                kernel_initializer=init,
                                padding='valid',
                                kernel_regularizer=_l2(weight_decay))(prev_x)
            if concat_x._keras_shape[1] != num_target_filters:
                concat_x = Convolution(\
                                filters=num_target_filters,
                                kernel_size=1,
                                ndim=ndim,
                                kernel_initializer=init,
                                padding='valid',
                                kernel_regularizer=_l2(weight_decay))(concat_x)
                
        #def _pad_to_fit(x, target_shape):
            #"""
            #Spatially pad a tensor's feature maps with zeros as evenly as
            #possible (center it) to fit the target shape.
            
            #Expected target shape is larger than the shape of the tensor.
            
            #NOTE: padding may be unequal on either side of the map if the
            #target dimension is odd. This is why keras's ZeroPadding2D isn't
            #used.
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
        #zero_pad = Lambda(_pad_to_fit,
                          #output_shape=concat_x._keras_shape[1:],
                          #arguments={'target_shape': concat_x.shape})
        #prev_x = zero_pad(prev_x)
        
        if merge_mode=='sum':
            merged = add([prev_x, concat_x])
        elif merge_mode=='concat':
            merged = concatenate([prev_x, concat_x], axis=1)
        else:
            raise ValueError("Unrecognized merge mode: {}"
                             "".format(merge_mode))
        return merged
    
    '''
    Build all the blocks on the contracting and expanding paths.
    '''
    tensors = {}
    model_input = Input(shape=input_shape)
    
    # Initial convolution
    x = Convolution(filters=input_num_filters,
                    kernel_size=3,
                    ndim=ndim,
                    kernel_initializer=init,
                    padding='same',
                    kernel_regularizer=_l2(weight_decay))(model_input)
    tensors[0] = x
    
    # DOWN (initial subsampling blocks)
    for b in range(0, num_init_blocks):
        depth = b+1
        x = residual_block(initblock,
                           nb_filter=nb_filter,
                           repetitions=1,
                           subsample=True,
                           **block_kwargs)(x)
        tensors[depth] = x
        v_print("INIT DOWN {}: {}".format(b, x._keras_shape))
    
    # DOWN (resnet blocks)
    for b in range(0, num_main_blocks):
        depth = b+1+num_init_blocks
        num_filters = input_num_filters*(2**b)
        x = residual_block(mainblock,
                           nb_filter=num_filters,
                           repetitions=get_repetitions(b),
                           subsample=True,
                           **block_kwargs)(x)
        v_print("MAIN DOWN {} (depth {}): {}".format( \
            depth, get_repetitions(b), x._keras_shape))
        
    # ACROSS
    num_filters = input_num_filters*(2**num_main_blocks)
    num_filters *= relative_num_across_filters
    x = residual_block(mainblock,
                       nb_filter=num_filters, 
                       repetitions=get_repetitions(num_main_blocks),
                       subsample=True,
                       upsample=True,
                       **block_kwargs)(x) 
    v_print("ACROSS (depth {}): {}".format( \
          get_repetitions(num_main_blocks), x._keras_shape))

    # UP (resnet blocks)
    for b in range(num_main_blocks-1, -1, -1):
        depth = b+1+num_init_blocks
        num_filters = input_num_filters*(2**b)
        if long_skip:
            num_across_filters = num_filters*relative_num_across_filters
            repetitions = get_repetitions(num_main_blocks)
            x = make_long_skip(prev_x=x,
                               concat_x=tensors[depth],
                               num_concat_filters=num_across_filters,
                               num_target_filters=num_filters,
                               **long_skip_kwargs)
        x = residual_block(mainblock,
                           nb_filter=num_filters,
                           repetitions=get_repetitions(b),
                           upsample=True,
                           **block_kwargs)(x)
        v_print("MAIN UP {} (depth {}): {}".format( \
            b, get_repetitions(b), x._keras_shape))
        
    # UP (final upsampling blocks)
    for b in range(num_init_blocks-1, -1, -1):
        depth = b+1
        if long_skip:
            num_across_filters = input_num_filters*relative_num_across_filters
            repetitions = get_repetitions(num_main_blocks)
            x = make_long_skip(prev_x=x,
                               concat_x=tensors[depth],
                               num_concat_filters=num_across_filters,
                               num_target_filters=input_num_filters,
                               **long_skip_kwargs)
        x = residual_block(initblock,
                           nb_filter=nb_filter,
                           repetitions=1,
                           upsample=True,
                           **block_kwargs)(x)
        v_print("INIT UP {}: {}".format(b, x._keras_shape))
        
    # Final convolution
    if long_skip:
        num_across_filters = input_num_filters*relative_num_across_filters
        repetitions = get_repetitions(num_main_blocks)
        x = make_long_skip(prev_x=x,
                           concat_x=tensors[0],
                           num_concat_filters=num_across_filters,
                           num_target_filters=input_num_filters,
                           **long_skip_kwargs)
    x = Convolution(filters=input_num_filters,
                    kernel_size=3,
                    ndim=ndim,
                    kernel_initializer=init,
                    padding='same',
                    kernel_regularizer=_l2(weight_decay))(x)
    
    if batch_norm:
        if bn_kwargs is None:
            bn_kwargs = {}
        x = BatchNormalization(axis=1, **bn_kwargs)(x)
    x = Activation('relu')(x)
    
    # OUTPUT (SOFTMAX)
    if num_classes is not None:
        all_outputs = []
        for i in range(num_outputs):
            # Linear classifier
            output = Convolution(filters=num_classes,
                                 kernel_size=1,
                                 ndim=ndim
                                 activation='linear',
                                 kernel_regularizer=_l2(weight_decay))(x)
            output = Permute((2,3,1))(output)
            if num_classes==1:
                output = Activation('sigmoid')(output)
            else:
                output = Activation(_softmax)(output)
            output = Permute((3,1,2))(output)
            all_outputs.append(output)
    else:
        # No classifier
        all_outputs = x
    
    # MODEL
    model = Model(inputs=model_input, outputs=output)

    return model
